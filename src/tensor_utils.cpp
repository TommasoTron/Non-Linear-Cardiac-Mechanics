#include "tensor_utils.hpp"

// TensorUtils: hyperelastic material via automatic differentiation (AD)
//given deformation gradient F, compute
//    P  = dW/dF   (first Piola-Kirchhoff stress tensor)
//    dP/dF        (consistent material tangent for Newton)
//
// The code defines a scalar strain-energy density W(F) and then utilises deal.II AD to obtain:
//   gradient(W) -> P,  hessian(W) -> dP/dF.


TensorUtils::TensorUtils() : ad_helper(dim * dim) {
      P_flat = Vector<double>(dim * dim);
      C_flat = FullMatrix<double>(dim * dim, dim * dim);
    }

void TensorUtils::compute_tensors(Tensor<2, dim> F, const Point<dim>& p, Tensor<2, dim> &P, Tensor<4, dim> &C) {


   const double d = 2.91;
  // const double z_base = 1.19;

  double x = p[0], y = p[1], z = p[2];
 


  Tensor<1, dim> f_vec; 
  f_vec[0] =1.0; f_vec[1] = 0.0; f_vec[2] = 0.0;

  Tensor<1, dim> s_vec;
  s_vec[0] = 0.0; s_vec[1] = 1.0; s_vec[2] = 0.0;

  Tensor<1, dim> n_vec;
  n_vec[0] = 0.0; n_vec[1] = 0.0; n_vec[2] = 1.0;



  ad_helper.reset();
  
      const FEValuesExtractors::Tensor<2> F_linearized(0);
      const FEValuesExtractors::Tensor<4> C_linearized(0);
      const typename AutoDiff::Types<ADNumberType>::tape_index tape_index = 0;
     
      // AD tape management:
        // On the first call, we "record" the computational graph of W(F).
  // On later calls we reuse the tape and only update the values of F.
      const bool is_recording =
          ad_helper.start_recording_operations(tape_index);
      if (is_recording) {
        // Register F as independent variables (inputs), compute W(F), and register W as dependent (output).
            // After recording, AD can produce derivatives of W with respect to entries of F.
        ad_helper.register_independent_variable(F, F_linearized);

        Tensor<2,dim,ADNumberType> F_AD = ad_helper.get_sensitive_variables(F_linearized);
        
        //ADNumberType W_ad = compute_W(F_AD);
        ADNumberType W_ad = compute_W(F_AD, f_vec, s_vec, n_vec);


        ad_helper.register_dependent_variable(W_ad);
        ad_helper.stop_recording_operations(false);
      } else {
        
        ad_helper.activate_recorded_tape(tape_index);
        ad_helper.set_independent_variable(F, F_linearized);
      }
      ad_helper.compute_gradient(P_flat);
      ad_helper.compute_hessian(C_flat);

      

      P = ad_helper.extract_gradient_component(P_flat, F_linearized);
      C = ad_helper.extract_hessian_component(C_flat, F_linearized,
                                           F_linearized);
    }

/*
typename TensorUtils::ADNumberType TensorUtils::compute_W(const Tensor<2, dim, ADNumberType> &F) const {
      const ADNumberType J_ad = determinant(F);
      const ADTensor2 C_ad = transpose(F) * F;
      const ADNumberType I1_ad = trace(C_ad);

      
      ADNumberType psi_ad =
          (mu_hook / 2.0) * (I1_ad * std::pow(J_ad, -2.0 / 3.0) - 3.0);
      psi_ad += (k_hook / 2.0) * std::pow(J_ad - 1.0, 2.0)
;

// //      using the isothropic formulation from the paper, pressure expressed in kPa
//       ADNumberType psi_ad = (a_iso / (2.0 * b_iso)) * (std::exp(b_iso * (I1_ad - 3.0)) - 1.0);

//       //adding this term as surrogate for orthothropic terms
//       psi_ad += (k_hook / 2.0) * std::pow(J_ad - 1.0, 2.0);
          
      return psi_ad;
    }
*/

typename TensorUtils::ADNumberType
TensorUtils::compute_W(const Tensor<2, dim, ADNumberType> &F,
                       const Tensor<1, dim> &f,
                       const Tensor<1, dim> &s,
                       const Tensor<1, dim> &n) const
{
  using ADTensor2 = Tensor<2, dim, ADNumberType>;

  // --- Kinematics ---
  const ADNumberType J_ad = determinant(F);
  const ADTensor2 C_ad = transpose(F) * F;

  // Green-Lagrange strain: E = 0.5 * (C - I)
  ADTensor2 I;
  for (unsigned int i = 0; i < dim; ++i) I[i][i] = 1.0;
  const ADTensor2 E_ad = 0.5 * (C_ad - I);

  // --- Project strain onto fiber coordinate system ---
  const ADNumberType E_ff = scalar_product(f, E_ad * f);
  const ADNumberType E_ss = scalar_product(s, E_ad * s);
  const ADNumberType E_nn = scalar_product(n, E_ad * n);

  const ADNumberType E_fs = scalar_product(f, E_ad * s);
  const ADNumberType E_fn = scalar_product(f, E_ad * n);
  const ADNumberType E_sn = scalar_product(s, E_ad * n);

  // --- Guccione exponential argument Q ---
const ADNumberType Q =
    b_ff  * E_ff * E_ff                    
  + b_sn  * (E_ss*E_ss + E_nn*E_nn + 2.0*E_sn*E_sn)  
  + b_fs * 2.0*(E_fs*E_fs + E_fn*E_fn);
  // --- Strain energy (Guccione) ---
  ADNumberType psi_ad = (C_param / 2.0) * (exp(Q) - 1.0);

  // --- Volumetric penalty (nearly incompressible) ---
  psi_ad += (k_hook / 2.0) * pow(J_ad - 1.0, 2.0);

  return psi_ad;
}


    
