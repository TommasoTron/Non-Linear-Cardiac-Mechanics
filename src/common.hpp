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

#include <iostream>

#ifndef NDEBUG
  #define check(condition, message)                                   \
    do {                                                              \
      if (!condition) {                                               \
        std::cerr << "[FAILED ASSERTION]" << std::endl;               \
        std::cerr << "  Condition: " << #condition << std::endl;      \
        std::cerr << "  Message: " << (message) << std::endl;         \
        std::cerr << "  File: " << __FILE__ << ", Line: " << __LINE__ \
                  << std::endl;                                       \
        std::cerr << "  Aborting..." << std::endl;                    \
        std::abort();                                                 \
      }                                                               \
    } while (0)

#else
  #define check(condition, message) ((void)0)
#endif

namespace AutoDiff = dealii::Differentiation::AD;

constexpr static AutoDiff::NumberTypes ADTypeCode =
    AutoDiff::NumberTypes::sacado_dfad_dfad;

constexpr static unsigned int dim = 3;
constexpr static unsigned int expected_dofs_per_cell = 30;

using VecADHelper = AutoDiff::VectorFunction<dim, ADTypeCode, double>;
using VecADNumberType = typename VecADHelper::ad_type;
using VFunctionWrapper = std::function<dealii::Tensor<1, dim, VecADNumberType>(
    dealii::Tensor<1, dim, VecADNumberType> &)>;
