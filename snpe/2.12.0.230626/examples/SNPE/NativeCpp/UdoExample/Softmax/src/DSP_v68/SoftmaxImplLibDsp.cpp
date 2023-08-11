//==============================================================================
// Auto Generated Code for SoftmaxUdoPackage
//==============================================================================

#include "optimize.h"
#include "op_register_ext.h"
#include "HTP/core/simple_reg.h"

BEGIN_PKG_OP_DEFINITION(PKG_Softmax)

// op execute function declarations
template <typename T>
int SoftmaxImpl(T &out, const T &in);

//op definitions
DEF_PACKAGE_OP((SoftmaxImpl<Tensor>), "Softmax")

/*
 * optimization definitions
 * need to be global in the package
 * one definition per optimization
 * syntax: DEF_PACKAGE_OPTIMIZATION(PRIORITY,MATCHCODE,CONSTRAINTCODE,REPLACECODE)
 * PRIORITY predefined values include EARLY(2000), MIDDLE(3000), LATE(4000)
 * HTP core provides some replacement functions for op package to use,
 * for more information about optimization rules, please refer to HTP core documentations
 */
// add code here

/* execute functions for ops */
template <typename T>
int SoftmaxImpl(T &out, const T &in) {

    out.set_dims(in);
    //NHWC
    auto [b_in, h_in, w_in, d_in] = in.dims();

    for (Idx b = 0; b < b_in; b++) {
        for (Idx h = 0; h < h_in; h++) {
            for (Idx w = 0; w < w_in; w++) {
                //Get maximum element
                float max = in(b, h, w, 0);
                for (Idx d = 0; d < d_in; d++) {
                    float inval = in(b, h, w, d);
                    max         = fmaxf(inval, max);
                }
                //Sum
                float sum = 0;
                for (Idx d = 0; d < d_in; d++) {
                    float inval = in(b, h, w, d);
                    sum += expf(inval - max);
                }
                //Normalization
                float sum_recip = 1.0f / sum;
                for (Idx d = 0; d < d_in; d++) {
                    float inval = in(b, h, w, d);
                    float outval = expf(inval - max);
                    out(b, h, w, d) = outval * sum_recip;
                }
            }
        }
    }
    return GraphStatus::Success;
}

END_PKG_OP_DEFINITION(PKG_Softmax)