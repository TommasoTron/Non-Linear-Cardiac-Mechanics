#include "left_ventricle.hpp"
#include <iostream>
#include <filesystem>


int main(int argc, char* argv[]){
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1); 
  //initialize MPI environment

// This program solves a *nonlinear* problem (cardiac-like geometry) in parallel.
// Pipeline:
//   1) setup():    read mesh, build FE space/DoFs, sparsity patter, vectors, matrices...
//   2) solve_newton(): repeatedly assemble residual+Jacobian and solve for updates
//   3) output():   write solution to VTU/PVTU for visualization
// MPI_InitFinalize keeps MPI alive for the entirity of main()

  namespace fs = std::filesystem;

  // Clean old solutions
  for (const auto& entry : fs::directory_iterator("../build/")) {
    std::string filename = entry.path().filename().string();
    if (filename.find("output-") == 0) {
      fs::remove(entry.path());
    }
  }

  // Remove old solution directories
  for (const auto& entry : fs::directory_iterator("../build/")) {
    std::string dirname = entry.path().filename().string();
    if (dirname.find("solution_") == 0 && fs::is_directory(entry.path())) {
      fs::remove_all(entry.path());
    }
  }

  // Get all .msh files in ../mesh/
  std::vector<std::string> mesh_files;
  for (const auto& entry : fs::directory_iterator("../mesh/")) {
    if (entry.path().extension() == ".msh") {
      mesh_files.push_back(entry.path().string());
    }
  }

  // For each mesh
  for (const auto& mesh : mesh_files) {
    std::string mesh_name = fs::path(mesh).stem().string();
    std::string subdir = "../build/solution_" + mesh_name;
    fs::create_directory(subdir);
    
    // Change to subdir
    fs::current_path(subdir);
    
    // Relative path to mesh from subdir
    std::string relative_mesh = "../../mesh/" + fs::path(mesh).filename().string();
    
    LV model = LV(relative_mesh, 2);
    model.setup();
    model.solve();
    
    // Change back to build
    fs::current_path("../../build");
  }

  
  
  return 0;
}
