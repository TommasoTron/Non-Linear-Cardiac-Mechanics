#include "left_ventricle.hpp"
#include "tensor_utils.hpp"

#include <cmath>
#include <Sacado.hpp>
#include <limits>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <deal.II/base/mpi.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/fe_field_function.h>
#include <deal.II/base/convergence_table.h>

void LV::setup() {
  pcout << "===============================================" << std::endl;

  // Create the mesh.

// read mesh, partition the serial mesh across MPI ranks
// created a distributed triangulation (mesh) used by DoFHandler/assembly
  {
    pcout << "Initializing the mesh" << std::endl;

    Triangulation<dim> mesh_serial;

    GridIn<dim> grid_in;
    grid_in.attach_triangulation(mesh_serial);

    std::ifstream grid_in_file(mesh_file_name);
    AssertThrow(grid_in_file.good(),
          ExcMessage("Could not open mesh file: '" + mesh_file_name +
                 "'. Current working directory is '" +
                 std::filesystem::current_path().string() + "'."));
    grid_in.read_msh(grid_in_file);

    GridTools::partition_triangulation(mpi_size, mesh_serial);
    const auto construction_data = TriangulationDescription::Utilities::
        create_description_from_triangulation(mesh_serial, MPI_COMM_WORLD);
    mesh.create_triangulation(construction_data);

    pcout << "  Number of elements = " << mesh.n_global_active_cells()
          << std::endl;

    // Debug: list boundary ids present in the mesh.
    //we had a lot of problems with boundary ids so this is useful to check them (2=dirichlet, 3=neumann,4=robin)
    {
      std::map<types::boundary_id, unsigned long long> boundary_face_counts;
      for (const auto &cell : mesh.active_cell_iterators()) {
        if (!cell->is_locally_owned())
          continue;
        if (!cell->at_boundary())
          continue;

        for (unsigned int f = 0; f < cell->n_faces(); ++f)
          if (cell->face(f)->at_boundary())
            ++boundary_face_counts[cell->face(f)->boundary_id()];
      }

      pcout << "  Boundary ids present (local face counts):";
      if (boundary_face_counts.empty())
        pcout << " <none>";
      for (const auto &[bid, cnt] : boundary_face_counts)
        pcout << " id=" << static_cast<unsigned int>(bid) << ":" << cnt;
      pcout << std::endl;
    }
  }
  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the finite element space. This is the same as in serial codes.
  {
    pcout << "Initializing the finite element space" << std::endl;

    fe = std::make_unique<FE_SimplexP<dim>>(r); //scalar element of degree r
    fs = std::make_unique<FESystem<dim>>(*fe, dim); //takes the same logic as the previous but vectorial 


    pcout << "  Degree                     = " << fe->degree << std::endl;
    pcout << "  DoFs per cell              = " << fs->dofs_per_cell
          << std::endl;

    check((fs->dofs_per_cell == expected_dofs_per_cell),
          "this needs to be fixed some how");

    quadrature = std::make_unique<QGaussSimplex<dim>>(fe->degree + 1);
    quadrature_face = std::make_unique<QGaussSimplex<dim - 1>>(fe->degree + 1);

    pcout << "  Quadrature points per cell = " << quadrature->size()
          << std::endl;
    pcout << "  Quadrature points per face = " << quadrature_face->size()
          << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the DoF handler.
  {
    pcout << "Initializing the DoF handler" << std::endl;

    dof_handler.reinit(mesh);
    dof_handler.distribute_dofs(*fs);

    // We retrieve the set of locally owned DoFs, which will be useful when
    // initializing linear algebra classes.
    locally_owned_dofs = dof_handler.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

    pcout << "  Number of DoFs = " << dof_handler.n_dofs() << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the linear system.
  {
    pcout << "Initializing the linear system" << std::endl;

    pcout << "  Initializing the sparsity pattern" << std::endl;

    // To initialize the sparsity pattern, we use Trilinos' class, that manages
    // some of the inter-process communication.
    TrilinosWrappers::SparsityPattern sparsity(locally_owned_dofs,
                                               MPI_COMM_WORLD);
    DoFTools::make_sparsity_pattern(dof_handler, sparsity);
    sparsity.compress();

    pcout << "  Initializing the Jacobian matrix" << std::endl;
    jacobian_matrix.reinit(sparsity); //allocate the jacobian matrix, will be used in the newton iteration

    pcout << "  Initializing the system right-hand side" << std::endl;
    system_rhs.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    pcout << "  Initializing the solution vector" << std::endl;
    solution_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    delta_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);

// solution is just a ghost vector used to evaluate gradients during assembly(robin term).
    solution.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);
  }
}


//assemble residual and jacobian for the current solution
//used in newton iteration
void LV::assemble_system() {

// Assemble residual R(u) and Jacobian J(u) = dR/du for the current solution vector.
  compute_rhs();
  //quite confident the jacobian is not 100% correctly assembled, but it still converges (quasi-newton)


// assemble each rank's contributions to the global system
jacobian_matrix.compress(VectorOperation::add);
system_rhs.compress(VectorOperation::add);




  std::map<types::global_dof_index, double> boundary_values;
  Functions::ZeroFunction<dim> zero_function(dim);


     VectorTools::interpolate_boundary_values(dof_handler,
                                              2,
                                              zero_function,
                                              boundary_values);

     MatrixTools::apply_boundary_values(
       boundary_values, jacobian_matrix, delta_owned, system_rhs, false); //false migliore in newton


}

void LV::compute_rhs() {
  pcout << "===============================================" << std::endl;

  pcout << "  Computing Right Hand Side" << std::endl;

  const unsigned int dofs_per_cell = fs->dofs_per_cell;
  const unsigned int n_q = quadrature->size();
  const unsigned int n_q_face = quadrature_face->size();


  //gives access to shape functions, gradients, JxW, at each quadrature point
  FEValues<dim> fe_values(*fs, *quadrature,
                          update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

  const FEValuesExtractors::Vector vec_index(0);

  FEFaceValues<dim> fe_face_values(*fs, *quadrature_face,
                                   update_values | update_normal_vectors |
                                       update_JxW_values | update_gradients | update_quadrature_points);

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double> cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  //reset global matrix and rhs
  jacobian_matrix = 0.0;
  system_rhs = 0.0;


    std::vector<double> solution_loc_face(n_q_face);
    std::vector<Tensor<2, dim>> solution_gradient_loc_face(n_q_face);
    std::vector<Tensor<2, dim>> solution_gradient_loc(n_q);
    //those vectors will hold the gradient of the current solution at quadrature points

    TensorUtils t_utils; //needed to compute P and dP/dF through automatic differentiation


  for (const auto &cell : dof_handler.active_cell_iterators()) {
    if (!cell->is_locally_owned())
      continue;

      //  binds FEValues to this specific cell and computes 
      //  all requested FE data at quadrature points.
    fe_values.reinit(cell);

    cell_rhs = 0.0;
    cell_matrix = 0.0;

    cell->get_dof_indices(dof_indices);

    fe_values[vec_index].get_function_gradients(solution,
                                                solution_gradient_loc);

    Tensor<1,expected_dofs_per_cell,VecADNumberType> local_solution;
    for (unsigned int i = 0; i < expected_dofs_per_cell; ++i)
      local_solution[i] = solution(dof_indices[i]);

    auto local_rhs_assembler =
      [&](Tensor<1,expected_dofs_per_cell,VecADNumberType>& local_u) {
        //volume quadrature loop
          for (unsigned int q = 0; q < n_q; ++q) {

            Tensor<2, dim, VecADNumberType> F;

            F.clear();// sanity check;
            for(unsigned int i = 0;i < dim; ++i)
              F[i][i] = VecADNumberType(1.0);
            

            //todo remove debugging. But it gave me so much headaches that I leave it for now
            for(unsigned int i = 0;i < dofs_per_cell; ++i){
              const Tensor<2, dim> grad_phi_i =
                  fe_values[vec_index].gradient(i, q);
               F += local_u[i] * grad_phi_i;
            }
            
            //debugging det(F)
            Tensor<2, dim> F_d;
            for (unsigned int ii = 0; ii < dim; ++ii)
              for (unsigned int jj = 0; jj < dim; ++jj)
                F_d[ii][jj] = Sacado::ScalarValue<VecADNumberType>::eval(F[ii][jj]);

            Tensor<2, dim> P;
            Tensor<4, dim> dP_dF;


            //for now it is implemented only the isotropic model (neo Hooke)
            //todo work on guccione anisotropic model


            // P is the PK stress tensor and is calculed by differentiating a neo-Hookean strain energy W(F)
            // dP_dF is the material elasticity tensor (4th order) needed for the jacobian assembly
            t_utils.compute_tensors(F_d, P, dP_dF); //computation through AD

            for (unsigned int i = 0; i < dofs_per_cell; ++i) {
              const Tensor<2, dim> grad_phi_i =
                  fe_values[vec_index].gradient(i, q);
              cell_rhs(i) += scalar_product(P, grad_phi_i) * fe_values.JxW(q);
              //internal residuals assembly

              for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                const Tensor<2, dim> grad_phi_j =
                    fe_values[vec_index].gradient(j, q);

                Tensor<2, dim> dP;
                dP.clear();
                for (unsigned int a = 0; a < dim; ++a)
                  for (unsigned int b = 0; b < dim; ++b)
                    for (unsigned int c = 0; c < dim; ++c)
                      for (unsigned int d = 0; d < dim; ++d)
                        dP[a][b] += dP_dF[a][b][c][d] * grad_phi_j[c][d];


                //newton jacobian assembly
                cell_matrix(i, j) +=
                    scalar_product(dP, grad_phi_i) * fe_values.JxW(q);
              }
            }
          }

          //quite unsure of these boundary terms, they were applied in the guccione example
          double pressure = 105.0; //in mmhg
          double alpha = 3.75;

           if (cell->at_boundary()) {
             for (unsigned int f = 0; f < cell->n_faces(); ++f) {
               if (cell->face(f)->at_boundary() &&
                   cell->face(f)->boundary_id() == 3) { //neumann BC
                 fe_face_values.reinit(cell, f);
                 fe_face_values[vec_index].get_function_gradients(
                     solution, solution_gradient_loc_face);

                 for (unsigned int q = 0; q < n_q_face; ++q) {
                   {
                     Tensor<2, dim> grad_u_face = solution_gradient_loc_face[q];

                     Tensor<2, dim> Fh = unit_symmetric_tensor<dim>();
          
                                Fh += grad_u_face;

                     const double det_Fh = determinant(Fh);
                     AssertThrow(std::isfinite(det_Fh) && det_Fh > 0.0,
                           ExcMessage(
                             "Non-positive or NaN det(Fh) on boundary face; cannot invert Fh."));

                     Tensor<2, dim> H = det_Fh * transpose(invert(Fh));

                     for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                       const Tensor<1, dim> term1 =
                           H * fe_face_values.normal_vector(q);
                       const unsigned int component_i =
                           fs->system_to_component_index(i).first;
                       const double phi_value = fe_face_values.shape_value(i, q);

                       cell_rhs(i) += term1[component_i] * phi_value *
                                      fe_face_values.JxW(q);
                     }
                   }
                 }
               }

               // ROBIN TERM
               
               
               if (cell->face(f)->at_boundary() && cell->face(f)->boundary_id()== 4) //changed robin bcs id
               {
                
               fe_face_values.reinit(cell,f);
       
                 fe_face_values.get_function_values(solution, solution_loc_face);
                 
                 for (unsigned int q=0; q<n_q_face;++q){
       
                 for (unsigned int i=0;i<dofs_per_cell;++i){
                   cell_rhs(i) += pressure*alpha * solution_loc_face[q] *
                   fe_face_values.shape_value(i, q) *
                                        fe_face_values.JxW(q);
                                      }
                                      
                                      for (unsigned int i=0;i<dofs_per_cell;++i){
                   for (unsigned int j=0;j<dofs_per_cell;++j){
                     cell_matrix(i,j)+= pressure*alpha*fe_face_values.shape_value(i,q)*
                     fe_face_values.shape_value(j,q)*
                     fe_face_values.JxW(q);
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
//    printf("the number of nonzero entries is %d\n", jacobian_matrix.n_nonzero_elements());
    system_rhs.add(dof_indices, cell_rhs);
  }
}
// removed commented out Dirichlet BCs which are not needed here. 
// they are applied in solve_newton instead

void LV::solve_linear_system() {

  //initial guess for the Newton increment.
  delta_owned = 0.0;
  delta_owned.compress(VectorOperation::insert);   //assigning the delta_owned vector to a Trilinos vector

  SolverControl solver_control(5000, 1.5e-2 * system_rhs.l2_norm());

  SolverGMRES<TrilinosWrappers::MPI::Vector> solver(solver_control);
  TrilinosWrappers::PreconditionAMG preconditioner;
  preconditioner.initialize(
      jacobian_matrix, TrilinosWrappers::PreconditionAMG::AdditionalData(1.0));

  solver.solve(jacobian_matrix, delta_owned, system_rhs, preconditioner);
  pcout << "  " << solver_control.last_step() << " GMRES iterations "
      << "(final residual " << std::scientific << std::setprecision(3)
      << solver_control.last_value() << ")" << std::endl;
        
}

LV::LineSearchResult
LV::line_search(const TrilinosWrappers::MPI::Vector &solution_prev,
                const TrilinosWrappers::MPI::Vector &delta_prev,
                const double residual_prev) {
  // Backtracking line search to avoid element inversion (det(F) <= 0) and to ensure the residual decreases.


  double alpha_ls = 1.0;
  const unsigned int max_backtracks = 25;
  const double rel_decrease_eps = 1e-8;
  const double abs_decrease_eps = 1e-14;


  const double alpha_min = 1e-7;

  bool last_trial_assembled = false;
  double last_trial_residual = std::numeric_limits<double>::infinity();

  TrilinosWrappers::MPI::Vector last_good_trial = solution_prev;
  double alpha_last_good = 0.0;

  for (unsigned int linesearch_iter = 0; linesearch_iter < max_backtracks;
       ++linesearch_iter) {
    TrilinosWrappers::MPI::Vector trial = solution_prev;
    trial.add(-alpha_ls, delta_prev);

    solution_owned = trial;
    solution = trial;

    bool assembly_ok = true;
    try {
      assemble_system();
    } catch (const dealii::ExceptionBase &) {
      assembly_ok = false;
    } catch (const std::exception &) {
      assembly_ok = false;
    }

    if (!assembly_ok) {
      solution_owned = solution_prev;
      solution = solution_prev;
      alpha_ls *= 0.5;
      continue;
    }

    last_trial_assembled = true;
    const double residual_trial = system_rhs.l2_norm();
    last_trial_residual = residual_trial;

    last_good_trial = trial;
    alpha_last_good = alpha_ls;

    // Accept if we get a meaningful decrease
    if (std::isfinite(residual_trial) &&
        (residual_trial < residual_prev * (1.0 - rel_decrease_eps) ||
         residual_trial < residual_prev - abs_decrease_eps)) {
      pcout << "  (alpha=" << std::scientific << std::setprecision(2) << alpha_ls
            << ")" << std::endl;
      return {true, false, alpha_ls, residual_trial};
    }
    alpha_ls *= 0.5;
  }

  // Not accepted
  if (last_trial_assembled && std::isfinite(last_trial_residual)) {
    // If we cannot decrease the residual and the step becomes tiny, we are in an infinite loop and that is pesos
    const double step_norm = alpha_last_good * delta_prev.l2_norm();
    const double sol_norm = solution_prev.l2_norm();
    const double step_tol = 1e-12 * (sol_norm + 1.0);

    if (step_norm <= step_tol || alpha_last_good <= alpha_min) {
      solution_owned = solution_prev;
      solution = solution_prev;
      pcout << "  (line search stagnated: alpha=" << std::scientific
            << std::setprecision(2) << alpha_last_good
            << ", step_norm=" << step_norm << ")" << std::endl;
      return {false, true, alpha_last_good, residual_prev};
    }

    // Otherwise, accept the smallest assembled step to avoid crashing,
    // but report it explicitly.
    solution_owned = last_good_trial;
    solution = last_good_trial;
    pcout << "  (alpha=" << std::scientific << std::setprecision(2)
          << alpha_last_good << ", no residual decrease)" << std::endl;
    return {false, false, alpha_last_good, last_trial_residual};
  }

  AssertThrow(false,
              ExcMessage(
                  "Line search failed: could not find a step length that keeps det(F)>0."));
  return {false, true, 0.0, residual_prev};
}


//solve the non linear system with newton's method + line search
void LV::solve_newton() {
  const unsigned int n_max_iters = 1000;
  const double residual_tolerance = 1e-5;

  unsigned int n_iter = 0;
  double residual_norm = residual_tolerance + 1;

  // We apply the boundary conditions to the initial guess (which is  stored in
  // solution_owned and solution).
  {
    std::set<types::boundary_id> dirichlet_boundary_id={2};
    ComponentMask dirichlet_mask =
        fs->component_mask(FEValuesExtractors::Vector(0)); //this is not actually necessary since we only have one component but now it never gives nans anymore

    IndexSet dirichlet_dofs =
        DoFTools::extract_boundary_dofs(dof_handler, dirichlet_mask,dirichlet_boundary_id);

    dirichlet_dofs = dirichlet_dofs & dof_handler.locally_owned_dofs();

    std::map<types::global_dof_index, double> boundary_values;
    //the boundary function is zero but this could work with any other function
    VectorTools::interpolate_boundary_values(dof_handler,
                                             2,
                                             function_g,
                                             boundary_values);

                                             

    for (const auto &idx : dirichlet_dofs)
      if (const auto it = boundary_values.find(idx);
          it != boundary_values.end())
      solution_owned[idx] = it->second;


    solution_owned.compress(VectorOperation::insert);
    solution = solution_owned;
  }

  

  bool stagnated = false;

  while (n_iter < n_max_iters && residual_norm > residual_tolerance) {
    assemble_system();
    residual_norm = system_rhs.l2_norm();

    //debugging
    AssertThrow(std::isfinite(residual_norm),
            ExcMessage("Newton residual is NaN/Inf"));


    pcout << "  Newton iteration " << n_iter << "/" << n_max_iters
          << " - ||r|| = " << std::scientific << std::setprecision(6)
          << residual_norm << std::flush;

    if (residual_norm > residual_tolerance) {
      solve_linear_system();

      const TrilinosWrappers::MPI::Vector solution_prev = solution_owned;
      const TrilinosWrappers::MPI::Vector delta_prev = delta_owned;
      const double residual_prev = residual_norm;

      const LineSearchResult ls = line_search(solution_prev, delta_prev, residual_prev);
      residual_norm = ls.residual;
      if (ls.stagnated) {
        stagnated = true;
        break;
      }
    } else {
      pcout << " < tolerance" << std::endl;
    }

    ++n_iter;
  }

  if (stagnated)
    pcout << "  Newton stopped due to stagnation." << std::endl;


  const bool converged = (!stagnated && residual_norm <= residual_tolerance);
  if (converged)
    pcout << "  Newton converged in " << n_iter << " iterations, final "
          << "residual ||r|| = " << std::scientific << std::setprecision(6)
          << residual_norm << std::endl;
  else
    pcout << "  Newton stopped in " << n_iter << " iterations, final "
          << "residual ||r|| = " << std::scientific << std::setprecision(6)
          << residual_norm << std::endl;
          
    }

 void
 LV::output() const //export solution to file
 {
   pcout << "===============================================" <<
   std::endl;

   IndexSet locally_relevant_dofs;
   DoFTools::extract_locally_relevant_dofs(dof_handler,
   locally_relevant_dofs);


   TrilinosWrappers::MPI::Vector solution_ghost(locally_owned_dofs,
                                                locally_relevant_dofs,
                                                MPI_COMM_WORLD);

//   relevant DoFs are received from other processes and stored inside solution_ghost.
   solution_ghost = solution;

   DataOut<dim> data_out;
   data_out.add_data_vector(dof_handler, solution_ghost, "solution");

   class SolutionMagnitudePostprocessor : public DataPostprocessorScalar<dim>
   {
   public:
     SolutionMagnitudePostprocessor()
       : DataPostprocessorScalar<dim>("solution_mag", update_values)
     {}

     void evaluate_vector_field(
       const DataPostprocessorInputs::Vector<dim> &inputs,
       std::vector<Vector<double>> &computed_quantities) const override
     {
       const unsigned int n_points = inputs.solution_values.size();

       for (unsigned int p = 0; p < n_points; ++p)
         {
           double sum_sq = 0.0;
           for (unsigned int d = 0; d < dim; ++d)
             sum_sq += inputs.solution_values[p][d] * inputs.solution_values[p][d];
           computed_quantities[p](0) = std::sqrt(sum_sq);
         }
     }
   };

   SolutionMagnitudePostprocessor magnitude_post;
   data_out.add_data_vector(dof_handler, solution_ghost, magnitude_post);

  std::vector<unsigned int> partition_int(mesh.n_active_cells());
   GridTools::get_subdomain_association(mesh, partition_int);
   const Vector<double> partitioning(partition_int.begin(),
   partition_int.end()); data_out.add_data_vector(partitioning,
   "partitioning");

   data_out.build_patches();

   const std::filesystem::path mesh_path(mesh_file_name);
   const std::string output_file_name = "output-" +
   mesh_path.stem().string();


   data_out.write_vtu_with_pvtu_record("./",
                                       output_file_name,
                                       0,
                                       MPI_COMM_WORLD);

   pcout << "Output written to " << output_file_name << std::endl;

   pcout << "===============================================" <<
   std::endl;
 }



 //trial to calculate difference between two solutions and plot them byut didn't work
 /*
double LV::compute_difference(const LV &reference,
                              const VectorTools::NormType norm) const
{

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
  // Expected pattern: ..._0_k.msh -> h = 0.k 
  // cube_0_2.msh  -> 0.2
  // cube_0_5.msh -> 0.5
  const std::string token = "_0_";
  const std::size_t pos   = mesh_file.rfind(token);
  AssertThrow(pos != std::string::npos,
              ExcMessage("Mesh filename does not contain '_0_': " + mesh_file));

  std::size_t i = pos + token.size();
  std::string digits;
  while (i < mesh_file.size() && std::isdigit(static_cast<unsigned char>(mesh_file[i])))
    {
      digits.push_back(mesh_file[i]);
      ++i;
    }

  AssertThrow(!digits.empty(),
              ExcMessage("Mesh filename has no digits after '_0_': " + mesh_file));

  return std::stod("0." + digits);
}

void LV::run_convergence_study(const std::vector<std::string> &mesh_files,
                               const unsigned int r,
                               const std::string &csv_filename)
{
  AssertThrow(mesh_files.size() >= 2,
              ExcMessage(
                "Need at least 2 meshes to compute self-convergence."));

  std::ofstream convergence_file(csv_filename);
  AssertThrow(convergence_file.good(),
              ExcMessage("Could not open '" + csv_filename + "'"));
  convergence_file << "h,diffL2,diffH1" << std::endl;

  dealii::ConvergenceTable table;

  std::vector<std::unique_ptr<LV>> models;
  models.reserve(mesh_files.size());

  for (const auto &mesh_file : mesh_files)
    {
      auto model = std::make_unique<LV>(mesh_file, r);
      model.setup();
      model.solve_newton();
      model.output();
      models.push_back(std::move(model));
    }

  // Compare each mesh solution against the next finer mesh.
  for (std::size_t i = 0; i + 1 < models.size(); ++i)
    {
      const double h = LV::h_from_mesh_filename(mesh_files[i]);

      const double diffL2 =
        models[i]->compute_difference(*models[i + 1], VectorTools::L2_norm);
      const double diffH1 =
        models[i]->compute_difference(*models[i + 1], VectorTools::H1_norm);

      table.add_value("h", h);
      table.add_value("L2", diffL2);
      table.add_value("H1", diffH1);

      convergence_file << h << "," << diffL2 << "," << diffH1 << std::endl;
    }

  table.evaluate_all_convergence_rates(ConvergenceTable::reduction_rate_log2);
  table.set_scientific("L2", true);
  table.set_scientific("H1", true);
  table.write_text(std::cout);
}
*/