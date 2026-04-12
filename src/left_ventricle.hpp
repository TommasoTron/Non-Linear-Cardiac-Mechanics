#pragma once

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values_extractors.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/differentiation/ad/ad_drivers.h>
#include <deal.II/differentiation/ad/ad_helpers.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <cmath>

#include "common.hpp"

using namespace dealii;


// Class managing the differential problem.
 
class LV {
public:
  // Physical dimension (1D, 2D, 3D)
  static constexpr unsigned int dim = 3;
  double compute_pressure(const Point<dim> &) const; //todo it will be interesting to compute pressure not as a constant but this should be done as a ultimate change

  // Self-convergence: compute ||u_this - u_reference|| in the chosen norm on this mesh.
  double compute_difference(const LV &,
                            const VectorTools::NormType norm) const;


  static void run_convergence_study(const std::vector<std::string> &mesh_files, //todo check if this convergence study works, I don't Remember
                                    const unsigned int r,
                                    const std::string &csv_filename =
                                      "convergence.csv");

  // Identifies geometry type; derived from mesh filename at construction.
  enum class MeshType { Beam, Ventricle };

  // Forcing term.
  class ForcingTerm : public Function<dim>{
  public:
    // Constructor.
    ForcingTerm() {}
  };

  // Dirichlet boundary conditions.
  class FunctionG : public Function<dim> {
  public:
    // Constructor.
    FunctionG() : Function<dim>(dim) {}

    // Evaluation of the boundary condition which in our setting is always going to be 0  (todo check if this is an overly semplification, I don't think it is)
    virtual double value(const Point<dim> & /*p*/, 
                         const unsigned int /*component*/ = 0) const override {
      return 0.0;
    }
  };

  FunctionG function_g;

  // Constructor. mesh_type and gmres_tol_factor are derived from the filename:
  //   filename contains "beam" → Beam  → gmres_tol_factor = 0.01
  //   otherwise               → Ventricle → gmres_tol_factor = 1.5e-4
  LV(const std::string &mesh_file_name_, const unsigned int &r_)
      : mesh_file_name(mesh_file_name_), r(r_),
        mesh_type(mesh_file_name_.find("beam") != std::string::npos
                  ? MeshType::Beam : MeshType::Ventricle),
        gmres_tol_factor(mesh_type == MeshType::Beam ? 0.01 : 1.5e-4),
        mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)),
        mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)),
        mesh(MPI_COMM_WORLD), pcout(std::cout, mpi_rank == 0) {}

  // Initialization.
  void setup();

  
  // System assembly.
  void assemble_system();

  // System solution.
  void solve_linear_system();

  void solve_newton();
  void solve();
  // Output.
  void output(unsigned int step) const;

protected:
  // Path to the mesh file.
  const std::string mesh_file_name;

  // Polynomial degree.
  const unsigned int r;

  // Geometry type, detected from mesh filename ("beam" → Beam, else Ventricle).
  MeshType mesh_type;

  double gmres_tol_factor; // derived from mesh_type: Beam→0.01, Ventricle→1.5e-4

  // Number of MPI processes.
  const unsigned int mpi_size;

  // This MPI process.
  const unsigned int mpi_rank;

  // TODO forcing term
  ForcingTerm forcing_term;

  double pressure;
  int last_iteration;
  
  // Triangulation. The parallel::fullydistributed::Triangulation class manages
  // a triangulation that is completely distributed (i.e. each process only
  // knows about the elements it owns and its ghost elements).
  parallel::fullydistributed::Triangulation<dim> mesh;

  // Finite element space.
  std::unique_ptr<FiniteElement<dim>> fe;

  // Finite System
  std::unique_ptr<FESystem<dim>> fs;

  // Quadrature formula.
  std::unique_ptr<Quadrature<dim>> quadrature;
  std::unique_ptr<Quadrature<dim - 1>> quadrature_face;

  // DoF handler.
  DoFHandler<dim> dof_handler;

  // System matrix.
  TrilinosWrappers::SparseMatrix system_matrix;

  // Jacobian Matrix
  TrilinosWrappers::SparseMatrix jacobian_matrix;

  // System right-hand side.
  TrilinosWrappers::MPI::Vector system_rhs;

  // System solution.
  TrilinosWrappers::MPI::Vector solution;

  // System solution (without ghost elements).
  TrilinosWrappers::MPI::Vector solution_owned;

  TrilinosWrappers::MPI::Vector delta_owned;

  // Parallel output stream.
  ConditionalOStream pcout;

  // DoFs owned by current process.
  IndexSet locally_owned_dofs;

  // DoFs needed to assemble locally owned cells (owned + ghosts).
  IndexSet locally_relevant_dofs;

  void compute_rhs();

  bool assembly_failed_local; // true if any rank-local det(F)<=0 during compute_rhs

  struct LineSearchResult {
    bool accepted;
    bool stagnated;
    double alpha;
    double residual;
  };

  LineSearchResult line_search(
      const TrilinosWrappers::MPI::Vector &solution_prev,
      const TrilinosWrappers::MPI::Vector &delta_prev,
      const double residual_prev);

  
      
/*
tentativo di convergenza ma non ha funzionato
static double h_from_mesh_filename(const std::string &mesh_file);
*/  


};
