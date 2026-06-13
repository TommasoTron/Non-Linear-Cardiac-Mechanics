#include "cardiac/lv.hpp"
#include "cardiac/strain_energy.hpp"
#include "cardiac/fiber_field.hpp"
#include "cardiac/Poisson.hpp"
#include "constants.hpp"

#include <deal.II/dofs/dof_tools.h>

std::unique_ptr<cardiac::LaplaceFibers> compute_laplace_fibers(cardiac::LV& model){
    DoFHandler<3> dof_handler_poisson;
  TrilinosWrappers::MPI::Vector lambda;

  Poisson poisson(model.get_mesh(), lambda,
                  model.get_fe(), model.get_quadrature(), model.get_quadrature_face(),
                  model.get_pcout());
  poisson.setup();
  poisson.assemble();
  poisson.solve();
  poisson.output();

  dof_handler_poisson.reinit(model.get_mesh());
  dof_handler_poisson.distribute_dofs(*model.get_fe());
  IndexSet locally_relevant;
  DoFTools::extract_locally_relevant_dofs(dof_handler_poisson, locally_relevant);

  TrilinosWrappers::MPI::Vector lambda_ghost(
      lambda.locally_owned_elements(), locally_relevant, MPI_COMM_WORLD);
  lambda_ghost = lambda;

  return std::make_unique<cardiac::LaplaceFibers>(
      dof_handler_poisson, lambda_ghost, *model.get_fe(), *model.get_quadrature());

}

int main(int argc, char *argv[]) {
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    LVBase::SolverParams params;
  
    if (argc > 1 && std::string(argv[1]) == "--convergence") {
    const std::vector<std::string> ventricle_meshes = {
      "../ventricular_meshes/msh/ventricle_0_99.msh",
      "../ventricular_meshes/msh/ventricle_0_7.msh"
      //      "../ventricular_meshes/msh/ventricle_0_5.msh",
      //"../ventricular_meshes/msh/ventricle_0_3.msh",
    };
    cardiac::LV::run_convergence_study(
        ventricle_meshes, 2,
        [](cardiac::LV& model) {     
          return std::make_pair(
              std::make_unique<cardiac::HolzapfelOgdenEnergy>(),
              compute_laplace_fibers(model));
        },
        params,
        cardiac::constants::holzapfel::CONVERGENCE_STUDY_STEPS,
        "convergence_ventricle.csv");
    return 0;
  }

  
  const std::string mesh = "../ventricular_meshes/msh/ventricle_0_99.msh";


  params.alpha_robin = cardiac::constants::holzapfel::ALPHA_ROBIN;

  cardiac::LV model(mesh, 2,
                    std::make_unique<cardiac::HolzapfelOgdenEnergy>(),
                    nullptr,
                    params);
  model.setup();
  
  model.set_fibers(compute_laplace_fibers(model));

  model.output_fibers("fibers_ventricle_laplace_0_7");
  
  model.solve();

  unsigned int n_ranks = dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
  model.save_solution("sol_" + std::to_string(n_ranks) + ".txt");
  return 0;
}
