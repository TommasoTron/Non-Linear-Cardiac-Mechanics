#pragma once

#include "lv_base.hpp"
#include "strain_energy.hpp"
#include "fiber_field.hpp"
#include "tensor_utils.hpp"

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace cardiac {

class LV : public LVBase {
public:
  LV(const std::string &mesh_file_name_,
     unsigned int r_,
     std::unique_ptr<StrainEnergy> energy,
     std::unique_ptr<FiberField>   fibers,
     const SolverParams &params_ = SolverParams{});

  void set_fibers(std::unique_ptr<FiberField> fibers) { fibers_ = std::move(fibers); }
  void set_energy(std::unique_ptr<StrainEnergy> energy) {energy_ = std::move(energy);}
  
  void solve();
  void solve(int num_steps);

  struct SerialReplica {
    Triangulation<dim>                  tria;
    std::unique_ptr<FiniteElement<dim>> fe;
    std::unique_ptr<FESystem<dim>>      fs;
    std::unique_ptr<MappingFE<dim>>     mapping;
    std::unique_ptr<Quadrature<dim>>    quadrature;
    DoFHandler<dim>                     dof_handler;
    Vector<double>                      solution;
  };

  std::unique_ptr<SerialReplica> gather_to_rank0() const;

  static double compute_difference_serial(const SerialReplica  &a,
                                          const SerialReplica  &b,
                                          VectorTools::NormType norm);

  static double h_from_mesh_filename(const std::string &mesh_file);

  double compute_difference(const LV &reference, VectorTools::NormType norm) const;

  using EnergyFiberFactory = std::function<std::pair<
    std::unique_ptr<StrainEnergy>,
    std::unique_ptr<FiberField>>(LV&)>;

  static void run_convergence_study(const std::vector<std::string> &mesh_files,
                                    unsigned int                    r,
                                    EnergyFiberFactory              make_ef,
                                    const SolverParams             &params,
                                    int                             n_steps,
                                    const std::string              &csv_filename
                                      = "convergence.csv");
  void output_fibers(const std::string &filename = "fibers") const;
  void save_solution(const std::string& filename) const;
protected:
  std::unique_ptr<StrainEnergy> energy_;
  std::unique_ptr<FiberField>   fibers_;

  void compute_rhs() override;
};

} // namespace cardiac
