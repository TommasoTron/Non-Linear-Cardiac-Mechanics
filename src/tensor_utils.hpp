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

#include "common.hpp"

// for reference:
// https://dealii.org/current/doxygen/deal.II/classDifferentiation_1_1AD_1_1ScalarFunction.html

using namespace dealii;

class TensorUtils {

  // TODO, import dim from LV
  constexpr static unsigned int dim = 3;
  using ADHelper = AutoDiff::ScalarFunction<dim, ADTypeCode, double>;
  using ADNumberType = typename ADHelper::ad_type;
  using ADTensor2 = Tensor<2, dim, ADNumberType>;

public:
  TensorUtils();

  void compute_tensors(Tensor<2, dim> F, Tensor<2, dim> &P, Tensor<4, dim> &C);

private:
  ADHelper ad_helper;
  Vector<double> P_flat;
  FullMatrix<double> C_flat;

  ADNumberType compute_W(const Tensor<2, dim, ADNumberType> &F) const;

  double mu_hook = 4.0;
  double k_hook = 2.0;
};
