#ifndef LV_HPP
#define LV_HPP

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_simplex_p.h>
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

#include <filesystem>
#include <fstream>
#include <iostream>

using namespace dealii;

/**
 * Class managing the differential problem.
 */
class LV
{
public:
  // Physical dimension (1D, 2D, 3D)
  static constexpr unsigned int dim = 3;



  struct input_data
  {
    

double b_f; //stretch in the fiber direction

double b_s; //stretch in the tangential direction

double b_n; //normal direction




    Tensor <1,dim> f0;  //vector in the fiber direction
    Tensor <1,dim> s0;  //vector in the sheet direction
    Tensor <1,dim> n0;  //vector in the normal direction
    

    //TODO add direction different from the 3 main ones 

    Tensor<2 ,dim> F; //deformation gradient

    double Ta; //active tension

    input_data( const Tensor<2,dim>& deformation_gradient_,
                const Tensor<1,dim>& fiber_direction_,
                const Tensor<1,dim>& sheet_direction_,
                const Tensor<1,dim>& normal_direction_,
                double active_tension,
                double bf,
                double bs,
                double bn):
                  F(deformation_gradient_)
                , f0(fiber_direction_)
                , s0(sheet_direction_)
                , n0(normal_direction_)
                , Ta(active_tension)
                , b_f(bf)
                , b_s(bs)
                , b_n(bn)
                {}

  };
  


  Tensor<2,dim> compute_P(input_data data) const;

  Tensor<4, dim> compute_dP_dF(const Vector<double> &d,
                                 const Point<dim> &p) const;


                                 // Forcing term.
  class ForcingTerm : public Function<dim> //TODO
  {
  public:
    // Constructor.
    ForcingTerm()
    {}

    
  };


  // Dirichlet boundary conditions.
  class FunctionG : public Function<dim>
  {
  public:
    // Constructor.
    FunctionG()
    {}

    // Evaluation.
    virtual double
    value(const Point<dim> & /*p*/,
          const unsigned int /*component*/ = 0) const override
    {
      return 0.0; //g(p[0]
    }
  };

using ADHelper = dealii::AutomaticDifferentiation::ADHelper<dim>; //dovrebbe permettere di calcolare la derivata in qualcge modo
ADHelper ad_helper;


  // Constructor.
  LV(const std::string &mesh_file_name_, const unsigned int &r_)
    : mesh_file_name(mesh_file_name_)
    , r(r_)
    , mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
    , mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
    , mesh(MPI_COMM_WORLD)
    , pcout(std::cout, mpi_rank == 0)
  {}

  // Initialization.
  void
  setup();

  // System assembly.
  void
  assemble();

  // System solution.
  void
  solve();

  // Output.
  void
  output() const;

protected:
  // Path to the mesh file.
  const std::string mesh_file_name;

  // Polynomial degree.
  const unsigned int r;

  // Number of MPI processes.
  const unsigned int mpi_size;

  // This MPI process.
  const unsigned int mpi_rank;



  // TODO forcing term
  ForcingTerm forcing_term;

  // TODO da fare il boundary condition
  FunctionG function_g;

  // Triangulation. The parallel::fullydistributed::Triangulation class manages
  // a triangulation that is completely distributed (i.e. each process only
  // knows about the elements it owns and its ghost elements).
  parallel::fullydistributed::Triangulation<dim> mesh;

  // Finite element space.
  std::unique_ptr<FiniteElement<dim>> fe;

  // Quadrature formula.
  std::unique_ptr<Quadrature<dim>> quadrature;

  // DoF handler.
  DoFHandler<dim> dof_handler;

  // System matrix.
  TrilinosWrappers::SparseMatrix system_matrix;

  // System right-hand side.
  TrilinosWrappers::MPI::Vector system_rhs;

  // System solution.
  TrilinosWrappers::MPI::Vector solution;

  // Parallel output stream.
  ConditionalOStream pcout;

  // DoFs owned by current process.
  IndexSet locally_owned_dofs;
};

#endif