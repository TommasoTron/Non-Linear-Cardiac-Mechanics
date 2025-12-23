#include "system_assembler.hpp"

template <unsigned int dim>
SystemAssembler<dim>::SystemAssembler(VFunctionWrapper &_fun)
    : fun(_fun), ad_helper(dim, dim) {}

template <unsigned int dim>
void SystemAssembler<dim>::compute_jacobian(FullMatrix<double> &mat,
                                            Tensor<1, dim, double> &u) {
  ad_helper.reset();

  const FEValuesExtractors::Tensor<1> u_linearized(0);

  const typename AutoDiff::Types<VecADNumberType>::tape_index tape_index = 1;

  const bool is_recording = ad_helper.start_recording_operations(tape_index);
  if (is_recording) {
    ad_helper.register_independent_variable(u, u_linearized);

    Tensor<1, dim, VecADNumberType> u_AD =
        ad_helper.get_sensitive_variables(u_linearized);

    Tensor<1, dim, VecADNumberType> rhs_AD = fun(u_AD);
    ad_helper.register_dependent_variable(rhs_AD, u_linearized);
    ad_helper.stop_recording_operations(false);

  } else {
    ad_helper.activate_recorded_tape(tape_index);
    ad_helper.set_independent_variable(u, u_linearized);
  }

  ad_helper.compute_jacobian(mat);
}
