#include "lv.hpp"
#include "../constants.hpp"

#include <Sacado.hpp>
#include <iomanip>
#include <limits>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <deal.II/base/mpi.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/fe_field_function.h>

namespace cardiac {

LV::LV(const std::string &mesh_file_name_,
       unsigned int r_,
       std::unique_ptr<StrainEnergy> energy,
       std::unique_ptr<FiberField>   fibers,
       const SolverParams &params_)
    : LVBase(mesh_file_name_, r_, params_)
    , energy_(std::move(energy))
    , fibers_(std::move(fibers))
{}

void LV::solve() { solve_loop(params.n_steps); }

void LV::solve(int num_steps) { solve_loop(num_steps); }

void LV::compute_rhs() {
  AssertThrow(energy_ != nullptr, ExcMessage("cardiac::LV: energy_ is null"));
  AssertThrow(fibers_ != nullptr, ExcMessage("cardiac::LV: fibers_ is null"));

  const unsigned int dofs_per_cell = fs->dofs_per_cell;
  const unsigned int n_q           = quadrature->size();
  const unsigned int n_q_face      = quadrature_face->size();

  FEValues<dim> fe_values(*fs, *quadrature,
                           update_values | update_gradients |
                           update_quadrature_points | update_JxW_values);

  const FEValuesExtractors::Vector vec_index(0);

  FEFaceValues<dim> fe_face_values(*fs, *quadrature_face,
                                    update_values | update_normal_vectors |
                                    update_JxW_values | update_gradients |
                                    update_quadrature_points);

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);
  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  jacobian_matrix = 0.0;
  system_rhs      = 0.0;

  std::vector<Tensor<1, dim>> solution_values_face(n_q_face);
  std::vector<Tensor<2, dim>> solution_gradient_loc_face(n_q_face);
  std::vector<Tensor<2, dim>> solution_gradient_loc(n_q);

  TensorUtils t_utils(*energy_);

  const double alpha_robin = params.alpha_robin;

  for (const auto &cell : dof_handler.active_cell_iterators()) {
    if (!cell->is_locally_owned())
      continue;

    fe_values.reinit(cell);

    cell_rhs    = 0.0;
    cell_matrix = 0.0;
    cell->get_dof_indices(dof_indices);

    fe_values[vec_index].get_function_gradients(solution, solution_gradient_loc);

    Tensor<1, expected_dofs_per_cell, VecADNumberType> local_solution;
    for (unsigned int i = 0; i < expected_dofs_per_cell; ++i)
      local_solution[i] = solution(dof_indices[i]);

    auto local_rhs_assembler =
      [&](Tensor<1, expected_dofs_per_cell, VecADNumberType> &local_u) {

        for (unsigned int q = 0; q < n_q; ++q) {
          Tensor<2, dim, VecADNumberType> F;
          F.clear();
          for (unsigned int i = 0; i < dim; ++i)
            F[i][i] = VecADNumberType(1.0);

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            F += local_u[i] * fe_values[vec_index].gradient(i, q);

          Tensor<2, dim> F_d;
          for (unsigned int ii = 0; ii < dim; ++ii)
            for (unsigned int jj = 0; jj < dim; ++jj)
              F_d[ii][jj] = Sacado::ScalarValue<VecADNumberType>::eval(F[ii][jj]);

          const double det_F = determinant(F_d);
          AssertThrow(std::isfinite(det_F) && det_F > 1e-12,
                      ExcMessage("Non-positive or near-singular det(F) in volume quadrature"));

          Tensor<2, dim> P;
          Tensor<4, dim> dP_dF;

          Tensor<1, dim> f0, s0, n0;
          fibers_->frame_at(cell, q, fe_values.quadrature_point(q), f0, s0, n0);

          t_utils.compute_tensors(F_d, P, dP_dF, f0, s0, n0);

          for (unsigned int i = 0; i < dofs_per_cell; ++i) {
            const Tensor<2, dim> grad_phi_i = fe_values[vec_index].gradient(i, q);
            cell_rhs(i) += scalar_product(P, grad_phi_i) * fe_values.JxW(q);

            for (unsigned int j = 0; j < dofs_per_cell; ++j) {
              const Tensor<2, dim> grad_phi_j = fe_values[vec_index].gradient(j, q);
              Tensor<2, dim> dP;
              dP.clear();
              for (unsigned int a = 0; a < dim; ++a)
                for (unsigned int b = 0; b < dim; ++b)
                  for (unsigned int c = 0; c < dim; ++c)
                    for (unsigned int d = 0; d < dim; ++d)
                      dP[a][b] += dP_dF[a][b][c][d] * grad_phi_i[c][d];

              cell_matrix(i, j) += scalar_product(dP, grad_phi_j) * fe_values.JxW(q);
            }
          }
        }

        if (cell->at_boundary()) {
          for (unsigned int f = 0; f < cell->n_faces(); ++f) {
            if (cell->face(f)->at_boundary() &&
                cell->face(f)->boundary_id() == constants::BID_PRESSURE) {
              fe_face_values.reinit(cell, f);
              fe_face_values[vec_index].get_function_gradients(solution,
                                                               solution_gradient_loc_face);

              for (unsigned int q = 0; q < n_q_face; ++q) {
                Tensor<2, dim> Fh;
                Fh.clear();
                for (unsigned int d = 0; d < dim; ++d)
                  Fh[d][d] = 1.0;
                Fh += solution_gradient_loc_face[q];

                const double det_Fh = determinant(Fh);
                AssertThrow(std::isfinite(det_Fh) && det_Fh > 0.0,
                            ExcMessage("Non-positive or NaN det(Fh) on boundary face"));

                const Tensor<2, dim> aux_F = transpose(invert(Fh));
                const Tensor<2, dim> H     = det_Fh * aux_F;

                for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                  const Tensor<1, dim> term1 = H * fe_face_values.normal_vector(q);
                  const Tensor<1, dim> phi_i = fe_face_values[vec_index].value(i, q);
                  cell_rhs(i) += pressure * scalar_product(term1, phi_i) * fe_face_values.JxW(q);

                  for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                    const Tensor<2, dim> grad_phi_j = fe_face_values[vec_index].gradient(j, q);
                    const double         prod        = scalar_product(grad_phi_j, aux_F);

                    Tensor<2, dim> term2;
                    for (unsigned int d = 0; d < dim; ++d)
                      term2[d][d] = prod;

                    const Tensor<2, dim> term3 = aux_F * transpose(grad_phi_j);
                    const Tensor<1, dim> term4 =
                        (term2 - term3) * (H * fe_face_values.normal_vector(q));

                    cell_matrix(i, j) +=
                        pressure * scalar_product(term4, phi_i) * fe_face_values.JxW(q);
                  }
                }
              }
            }

            if (cell->face(f)->at_boundary() &&
                cell->face(f)->boundary_id() == constants::BID_ROBIN) {
              fe_face_values.reinit(cell, f);
              fe_face_values[vec_index].get_function_values(solution, solution_values_face);

              for (unsigned int q = 0; q < n_q_face; ++q) {
                const Tensor<1, dim> u_q = solution_values_face[q];
                for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                  const Tensor<1, dim> phi_i = fe_face_values[vec_index].value(i, q);
                  cell_rhs(i) += alpha_robin * scalar_product(u_q, phi_i) * fe_face_values.JxW(q);

                  for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                    const Tensor<1, dim> phi_j = fe_face_values[vec_index].value(j, q);
                    cell_matrix(i, j) +=
                        alpha_robin * scalar_product(phi_i, phi_j) * fe_face_values.JxW(q);
                  }
                }
              }
            }
          }
        }
      };

    local_rhs_assembler(local_solution);

    cell->get_dof_indices(dof_indices);
    jacobian_matrix.add(dof_indices, cell_matrix);
    system_rhs.add(dof_indices, cell_rhs);
  }
}

double LV::compute_difference(const LV &reference, VectorTools::NormType norm) const
{
  AssertThrow(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) == 1,
              ExcMessage("compute_difference requires a single MPI rank "
                         "(FEFieldFunction does not support distributed meshes)."));

  MappingFE<dim> mapping(*fe);

  Functions::FEFieldFunction<dim, TrilinosWrappers::MPI::Vector> u_ref(
    reference.dof_handler, reference.solution, mapping);

  Vector<double> difference_per_cell(mesh.n_active_cells());
  VectorTools::integrate_difference(mapping,
                                    dof_handler,
                                    solution,
                                    u_ref,
                                    difference_per_cell,
                                    *quadrature,
                                    norm);

  return VectorTools::compute_global_error(mesh, difference_per_cell, norm);
}

double LV::h_from_mesh_filename(const std::string &mesh_file)
{
  const std::filesystem::path p(mesh_file);
  const std::string stem = p.stem().string();

  std::string digits;
  for (auto it = stem.rbegin(); it != stem.rend(); ++it) {
    if (std::isdigit(static_cast<unsigned char>(*it)))
      digits.insert(digits.begin(), *it);
    else if (*it == '_' && !digits.empty())
      continue;
    else
      break;
  }

  AssertThrow(!digits.empty(),
              ExcMessage("Mesh filename has no trailing digits: " + mesh_file));

  std::size_t first_non_zero = digits.find_first_not_of('0');
  if (first_non_zero == std::string::npos)
    digits = "0";
  else
    digits = digits.substr(first_non_zero);

  return std::stod("0." + digits);
}

std::unique_ptr<LV::SerialReplica> LV::gather_to_rank0() const
{
  using CellPayload = std::pair<CellId, std::vector<double>>;
  std::vector<CellPayload> local_data;

  std::vector<types::global_dof_index> dof_indices(fs->dofs_per_cell);
  for (const auto &cell : dof_handler.active_cell_iterators()) {
    if (!cell->is_locally_owned())
      continue;
    cell->get_dof_indices(dof_indices);
    std::vector<double> values(fs->dofs_per_cell);
    for (unsigned int i = 0; i < fs->dofs_per_cell; ++i)
      values[i] = solution(dof_indices[i]);
    local_data.emplace_back(cell->id(), std::move(values));
  }

  const auto all_data = Utilities::MPI::gather(MPI_COMM_WORLD, local_data, 0);

  if (mpi_rank != 0)
    return nullptr;

  std::map<CellId, std::vector<double>> by_cell;
  for (const auto &per_rank : all_data)
    for (const auto &payload : per_rank)
      by_cell[payload.first] = payload.second;

  auto rep = std::make_unique<SerialReplica>();
  rep->fe         = std::make_unique<FE_SimplexP<dim>>(r);
  rep->fs         = std::make_unique<FESystem<dim>>(*rep->fe, dim);
  rep->mapping    = std::make_unique<MappingFE<dim>>(*rep->fe);
  rep->quadrature = std::make_unique<QGaussSimplex<dim>>(rep->fe->degree + 1);

  GridIn<dim> grid_in;
  grid_in.attach_triangulation(rep->tria);
  std::ifstream grid_in_file(mesh_file_name);
  AssertThrow(grid_in_file.good(),
              ExcMessage("gather_to_rank0: could not open mesh file '" +
                         mesh_file_name + "'"));
  grid_in.read_msh(grid_in_file);

  rep->dof_handler.reinit(rep->tria);
  rep->dof_handler.distribute_dofs(*rep->fs);
  rep->solution.reinit(rep->dof_handler.n_dofs());

  std::vector<types::global_dof_index> serial_dof_indices(rep->fs->dofs_per_cell);
  for (const auto &cell : rep->dof_handler.active_cell_iterators()) {
    const auto it = by_cell.find(cell->id());
    AssertThrow(it != by_cell.end(),
                ExcMessage("gather_to_rank0: missing cell " + cell->id().to_string()));
    cell->get_dof_indices(serial_dof_indices);
    for (unsigned int i = 0; i < rep->fs->dofs_per_cell; ++i)
      rep->solution(serial_dof_indices[i]) = it->second[i];
  }

  return rep;
}

double LV::compute_difference_serial(const SerialReplica  &a,
                                     const SerialReplica  &b,
                                     VectorTools::NormType norm)
{
  AssertThrow(norm == VectorTools::L2_norm || norm == VectorTools::H1_norm,
              ExcMessage("compute_difference_serial supports only L2 and H1 norms."));

  Functions::FEFieldFunction<dim, Vector<double>> u_b(b.dof_handler, b.solution, *b.mapping);

  const FEValuesExtractors::Vector vec_index(0);
  const unsigned int n_q = a.quadrature->size();

  UpdateFlags flags = update_values | update_quadrature_points | update_JxW_values;
  if (norm == VectorTools::H1_norm)
    flags |= update_gradients;

  FEValues<dim> fe_values_a(*a.mapping, *a.fs, *a.quadrature, flags);

  std::vector<Tensor<1, dim>> u_a_values(n_q);
  std::vector<Tensor<2, dim>> u_a_grads(n_q);

  double       sum_l2_sq          = 0.0;
  double       sum_h1_seminorm_sq = 0.0;
  unsigned int total_qp           = 0;
  unsigned int skipped_qp         = 0;

  for (const auto &cell : a.dof_handler.active_cell_iterators()) {
    fe_values_a.reinit(cell);
    fe_values_a[vec_index].get_function_values(a.solution, u_a_values);
    if (norm == VectorTools::H1_norm)
      fe_values_a[vec_index].get_function_gradients(a.solution, u_a_grads);

    for (unsigned int q = 0; q < n_q; ++q) {
      ++total_qp;
      const Point<dim> &p = fe_values_a.quadrature_point(q);

      Vector<double>              u_b_val(dim);
      std::vector<Tensor<1, dim>> u_b_grad_vec(dim);
      bool found = true;

      try {
        u_b.vector_value(p, u_b_val);
        if (norm == VectorTools::H1_norm)
          u_b.vector_gradient(p, u_b_grad_vec);
      } catch (const dealii::ExceptionBase &) {
        found = false;
      }

      if (!found) {
        ++skipped_qp;
        continue;
      }

      double diff_sq = 0.0;
      for (unsigned int d = 0; d < dim; ++d) {
        const double diff_d = u_a_values[q][d] - u_b_val[d];
        diff_sq += diff_d * diff_d;
      }
      sum_l2_sq += diff_sq * fe_values_a.JxW(q);

      if (norm == VectorTools::H1_norm) {
        double grad_diff_sq = 0.0;
        for (unsigned int d = 0; d < dim; ++d)
          for (unsigned int e = 0; e < dim; ++e) {
            const double g_de = u_a_grads[q][d][e] - u_b_grad_vec[d][e];
            grad_diff_sq += g_de * g_de;
          }
        sum_h1_seminorm_sq += grad_diff_sq * fe_values_a.JxW(q);
      }
    }
  }

  if (skipped_qp > 0)
    std::cout << "  [compute_difference_serial] skipped "
              << skipped_qp << " of " << total_qp
              << " quadrature points (boundary mismatch on curved geometry)"
              << std::endl;

  if (norm == VectorTools::L2_norm)
    return std::sqrt(sum_l2_sq);
  else
    return std::sqrt(sum_l2_sq + sum_h1_seminorm_sq);
}

void LV::run_convergence_study(const std::vector<std::string> &mesh_files,
                               unsigned int r,
                               EnergyFiberFactory make_ef,
                               const SolverParams &params,
                               int n_steps,
                               const std::string &csv_filename)
{
  AssertThrow(mesh_files.size() >= 2,
              ExcMessage("Need at least 2 meshes to compute self-convergence."));

  const unsigned int rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  ConditionalOStream pcout(std::cout, rank == 0);

  std::ofstream convergence_file;
  if (rank == 0) {
    convergence_file.open(csv_filename);
    AssertThrow(convergence_file.good(),
                ExcMessage("Could not open '" + csv_filename + "'"));
    convergence_file << "h,diffL2,rateL2,diffH1,rateH1" << std::endl;
  }

  std::vector<std::unique_ptr<SerialReplica>> replicas;
  if (rank == 0)
    replicas.reserve(mesh_files.size());

  for (const auto &mesh_file : mesh_files) {
    pcout << "Convergence study: solving on " << mesh_file << "\n" << std::endl;
    auto model = std::make_unique<LV>(mesh_file, r, nullptr, nullptr, params);
    model->setup();
    auto [energy, fibers] = make_ef(*model);
    model->set_fibers(std::move(fibers));
    model->set_energy(std::move(energy));
    model->solve(n_steps);
    auto rep = model->gather_to_rank0();
    if (rank == 0)
      replicas.push_back(std::move(rep));
  }

  if (rank != 0)
    return;

  std::vector<double> hs, l2s, h1s;
  for (std::size_t i = 0; i + 1 < replicas.size(); ++i) {
    hs.push_back(LV::h_from_mesh_filename(mesh_files[i]));
    l2s.push_back(compute_difference_serial(*replicas[i], *replicas[i + 1], VectorTools::L2_norm));
    h1s.push_back(compute_difference_serial(*replicas[i], *replicas[i + 1], VectorTools::H1_norm));
  }

  auto rate = [](double err_prev, double err_curr, double h_prev, double h_curr) -> double {
    return std::log(err_prev / err_curr) / std::log(h_prev / h_curr);
  };

  std::cout << "\nConvergence table\n" << std::endl;
  std::cout << std::left
            << std::setw(8)  << "h"
            << std::setw(14) << "L2"
            << std::setw(10) << "rate_L2"
            << std::setw(14) << "H1"
            << std::setw(10) << "rate_H1"
            << std::endl;

  for (std::size_t i = 0; i < hs.size(); ++i) {
    double rL2 = std::numeric_limits<double>::quiet_NaN();
    double rH1 = std::numeric_limits<double>::quiet_NaN();
    if (i > 0) {
      rL2 = rate(l2s[i - 1], l2s[i], hs[i - 1], hs[i]);
      rH1 = rate(h1s[i - 1], h1s[i], hs[i - 1], hs[i]);
    }

    std::cout << std::left
              << std::fixed      << std::setprecision(4) << std::setw(8)  << hs[i]
              << std::scientific << std::setprecision(4) << std::setw(14) << l2s[i];
    if (i == 0) std::cout << std::setw(10) << "-";
    else        std::cout << std::fixed << std::setprecision(2) << std::setw(10) << rL2;
    std::cout << std::scientific << std::setprecision(4) << std::setw(14) << h1s[i];
    if (i == 0) std::cout << std::setw(10) << "-";
    else        std::cout << std::fixed << std::setprecision(2) << std::setw(10) << rH1;
    std::cout << std::endl;

    convergence_file << hs[i] << "," << l2s[i] << ","
                     << (i == 0 ? std::string("") : std::to_string(rL2)) << ","
                     << h1s[i] << ","
                     << (i == 0 ? std::string("") : std::to_string(rH1))
                     << std::endl;
  }

  
}

void LV::output_fibers(const std::string &filename) const {
  AssertThrow(fibers_ != nullptr, ExcMessage("Fibers not initialized."));

  FE_SimplexDGP<dim> fe_scalar(0);
  FESystem<dim> fe_vec(fe_scalar, 3);
  DoFHandler<dim> dof_vec(mesh);
  dof_vec.distribute_dofs(fe_vec);

  const IndexSet locally_owned_vec = dof_vec.locally_owned_dofs();
  IndexSet locally_relevant_vec;
  DoFTools::extract_locally_relevant_dofs(dof_vec, locally_relevant_vec);

  TrilinosWrappers::MPI::Vector f0_owned(locally_owned_vec, MPI_COMM_WORLD);
  TrilinosWrappers::MPI::Vector s0_owned(locally_owned_vec, MPI_COMM_WORLD);
  TrilinosWrappers::MPI::Vector n0_owned(locally_owned_vec, MPI_COMM_WORLD);

  std::vector<types::global_dof_index> dofs(3);
  auto cell_vec = dof_vec.begin_active();
  for (const auto &cell : dof_handler.active_cell_iterators()) {
    if (cell->is_locally_owned()) {
      dealii::Tensor<1, 3> f0, s0, n0;
      fibers_->frame_at(cell, 0, cell->center(), f0, s0, n0);
      cell_vec->get_dof_indices(dofs);
      for (unsigned int i = 0; i < 3; ++i) {
        f0_owned[dofs[i]] = f0[i];
        s0_owned[dofs[i]] = s0[i];
        n0_owned[dofs[i]] = n0[i];
      }
    }
    ++cell_vec;
  }
  f0_owned.compress(VectorOperation::insert);
  s0_owned.compress(VectorOperation::insert);
  n0_owned.compress(VectorOperation::insert);

  TrilinosWrappers::MPI::Vector f0_v(locally_owned_vec, locally_relevant_vec, MPI_COMM_WORLD);
  TrilinosWrappers::MPI::Vector s0_v(locally_owned_vec, locally_relevant_vec, MPI_COMM_WORLD);
  TrilinosWrappers::MPI::Vector n0_v(locally_owned_vec, locally_relevant_vec, MPI_COMM_WORLD);
  f0_v = f0_owned;
  s0_v = s0_owned;
  n0_v = n0_owned;

  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_vec);

  const std::vector<DataComponentInterpretation::DataComponentInterpretation>
    vi(3, DataComponentInterpretation::component_is_part_of_vector);

  data_out.add_data_vector(f0_v, std::vector<std::string>(3, "f0"),
                           DataOut<dim>::type_dof_data, vi);
  data_out.add_data_vector(s0_v, std::vector<std::string>(3, "s0"),
                           DataOut<dim>::type_dof_data, vi);
  data_out.add_data_vector(n0_v, std::vector<std::string>(3, "n0"),
                           DataOut<dim>::type_dof_data, vi);

  std::vector<unsigned int> partition_int(mesh.n_active_cells());
  GridTools::get_subdomain_association(mesh, partition_int);
  const Vector<double> partitioning(partition_int.begin(), partition_int.end());
  data_out.add_data_vector(partitioning, "partitioning");

  data_out.build_patches();
  data_out.write_vtu_with_pvtu_record("./", filename, 0, MPI_COMM_WORLD);
}

void LV::save_solution(const std::string &filename) const
{

  auto rep = gather_to_rank0();

  if (mpi_rank == 0 && rep != nullptr) {
    std::ofstream out(filename);

    out << std::scientific << std::setprecision(16);

    for (unsigned int i = 0; i < rep->solution.size(); ++i) {
      out << rep->solution(i) << "\n";
    }
    
    std::cout << "exported sol to: " << filename << std::endl;
  }
}

} // namespace cardiac
