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


#include <memory>
#include <functional>

#include "common.hpp"

// for reference:
// https://dealii.org/current/doxygen/deal.II/classDifferentiation_1_1AD_1_1VectorFunction.html

using namespace dealii;

template<unsigned int dim>
class SystemAssembler {
 
public:
  SystemAssembler(VFunctionWrapper& _fun);
  void compute_jacobian(FullMatrix<double>& Mat, Tensor<1,dim,double>& u);

private:
  VFunctionWrapper& fun;
  VecADHelper ad_helper;
};
