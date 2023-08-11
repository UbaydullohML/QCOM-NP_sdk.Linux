//==============================================================================
// Auto Generated Code for Conv2DPackage
//==============================================================================
#include "optimize.h"
#include "op_register_ext.h"
#include "HTP/core/simple_reg.h"

BEGIN_PKG_OP_DEFINITION(PKG_Conv)

// op execute function declarations
template<typename TensorType>
int convImpl(TensorType& out_0,
             const TensorType& in_0,
             const TensorType& weight,
             const TensorType& bias,
             const Tensor& group,
             const Tensor& pads,
             const Tensor& strides,
             const Tensor& dilations,
             const Tensor& kernel_shape);

//op definitions
DEF_PACKAGE_OP((convImpl<Tensor>), "Conv")

/* execute functions for ops */

template<typename TensorType>
int convImpl(TensorType& out_0,
             const TensorType& in_0,
             const TensorType& weight,
             const TensorType& bias,
             const Tensor& group,
             const Tensor& pads,
             const Tensor& strides,
             const Tensor& dilations,
             const Tensor& kernel_shape)

{
    //Initialise params
    int32_t groups = group(0, 0, 0, 0);
    int32_t padH = pads(0, 0, 0, 0);
    int32_t padW = pads(0, 0, 0, 1);
    int32_t strideH = strides(0, 0, 0, 0);
    int32_t strideW = strides(0, 0, 0, 1);

    auto [b_out, h_out, w_out, d_out] = out_0.dims();

    int32_t d_filter = weight.dim(1);
    int32_t h_filter = weight.dim(2);
    int32_t w_filter = weight.dim(3);

    int32_t h_in = in_0.dim(1);
    int32_t w_in = in_0.dim(2);

    Idx outputGroupDepth = d_out / groups;

    for (int32_t ob = 0; ob < b_out; ob++)
    {
        for (int32_t oh = 0; oh < h_out; oh++)
        {
            for (int32_t ow = 0; ow < w_out; ow++)
            {
                for (int32_t g = 0; g < groups; g++)
                {
                    for (int32_t d = 0; d < outputGroupDepth; d++)
                    {
                        int32_t inputOriginH = (int32_t) oh * strideH - padH;
                        int32_t inputOriginW = (int32_t) ow * strideW - padW;

                        float sum = 0.0f;
                        int32_t depth = d + g * outputGroupDepth;
                        sum += bias(0, 0, 0, depth);
                        for (uint32_t fh = 0; fh < h_filter; fh++)
                        {
                            for (uint32_t fw = 0; fw < w_filter; fw++)
                            {
                                int32_t inputH  = inputOriginH + (int32_t) fh;
                                int32_t inputW  = inputOriginW + (int32_t) fw;
                                for (uint32_t fd = 0; fd < d_filter; fd++)
                                {
                                    if (inputH >= 0 && inputH < (int32_t)(h_in) && inputW >= 0 &&
                                        inputW < (int32_t)(w_in))
                                    {
                                        float inval = in_0(ob, inputH, inputW, fd);
                                        float filtval = weight(depth, fd, fh, fw);
                                        sum += inval * filtval;
                                    }

                                }
                            }
                        }

                        out_0(ob, oh, ow, depth) = sum;
                    }
                }
            }
        }
    }

    return GraphStatus::Success;
}


END_PKG_OP_DEFINITION(PKG_Conv)
