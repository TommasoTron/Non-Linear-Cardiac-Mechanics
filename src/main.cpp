#include "left_ventricle.hpp"
#include <iostream>


int main(int argc, char* argv[]){
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1); 
  //initialize MPI environment

// This program solves a *nonlinear* problem (cardiac-like geometry) in parallel.
// Pipeline:
//   1) setup():    read mesh, build FE space/DoFs, sparsity patter, vectors, matrices...
//   2) solve_newton(): repeatedly assemble residual+Jacobian and solve for updates
//   3) output():   write solution to VTU/PVTU for visualization
// MPI_InitFinalize keeps MPI alive for the entirity of main()


  std::string ventricular_mesh_path = "../ventricular_meshes/msh/ventricle_0_7.msh";
 // ventricular meshes created with geo file in mesh/ventricular_meshes/geo
  
  std::string cubic_mesh_path = "../mesh/cube_0_2.msh";

  


  LV model = LV(ventricular_mesh_path, 2); //2 is the polynomial degree
  //todo could study how the degree affects the solution


  model.setup();


  model.solve_newton(); //solve the nonlinear problem by (pseudo)Newton+line search

  model.output(); //export solution displacement field
  
  
  return 0;
}
