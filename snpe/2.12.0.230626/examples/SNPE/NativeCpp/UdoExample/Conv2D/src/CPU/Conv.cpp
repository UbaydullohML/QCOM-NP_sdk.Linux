//==============================================================================
// Auto Generated Code for Conv2DPackage
//==============================================================================
#include <iostream>
#include <string>
#include <string.h>
#include <cmath>
#include <algorithm>

#include "CpuBackendUtils.hpp"
#include "CustomOpPackage.hpp"

using namespace qnn::custom;
using namespace qnn::custom::utils;

namespace conv {

Qnn_ErrorHandle_t execute(CustomOp* operation) {
    int32_t groups = 1;
    int32_t* pad = nullptr;
    int32_t padH = 0;
    int32_t padW = 0;
    int32_t* stride = nullptr;
    int32_t strideH = 1;
    int32_t strideW = 1;
    int32_t* dilation = nullptr;
    int32_t* kernel_shape = nullptr;

    auto m_Inputs = operation->getInput(0);
    auto m_Outputs = operation->getOutput(0);

    const float* in  = (float*)m_Inputs->data;
    float* out  = (float*)m_Outputs->data;

    float* filter  = (float*)(operation->getInput(1))->data;
    float* bias  = (float*)(operation->getInput(2))->data;

    groups = (int32_t)(operation->getParam("group")->scalarParam);
    pad = ((int32_t*)(operation->getParam("pads")->tensorParam->data));
    stride = ((int32_t*)(operation->getParam("strides")->tensorParam->data));
    dilation = ((int32_t*)(operation->getParam("dilations")->tensorParam->data));
    kernel_shape = ((int32_t*)(operation->getParam("kernel_shape")->tensorParam->data));

    if (pad != nullptr)
    {
        padH = pad[0];
        padW = pad[1];
    }

    if (stride != nullptr)
    {
        strideH = stride[0];
        strideW = stride[1];
    }

    //Input height, width and depth.
    int32_t inputHeight = m_Inputs->currentDimensions[1];
    int32_t inputWidth = m_Inputs->currentDimensions[2];
    int32_t inputDepth = m_Inputs->currentDimensions[3];

    //Output height, width and depth
    int32_t outputHeight = m_Outputs->currentDimensions[1];
    int32_t outputWidth = m_Outputs->currentDimensions[2];
    int32_t outputDepth = m_Outputs->currentDimensions[3];

    //Filter height, width and depth
    int32_t filterHeight  = (operation->getInput(1))->currentDimensions[0];
    int32_t filterWidth = (operation->getInput(1))->currentDimensions[1];
    int32_t filterDepth = (operation->getInput(1))->currentDimensions[2];

    // set the depth for each group of filters
    int32_t outputGroupDepth = outputDepth / groups;

    float outputActivationMin = std::numeric_limits<float>::lowest();
    float outputActivationMax = std::numeric_limits<float>::max();
    for(int32_t oh = 0; oh < outputHeight; oh++) {
       for(int32_t ow = 0; ow < outputWidth; ow++) {
          for (int32_t g = 0; g < groups; g++) {
              for (int32_t d = 0; d < outputGroupDepth; d++) {
                  int offset = g * outputGroupDepth + d;
                  float sum = 0.0f;
                  for(int32_t fh = 0; fh < filterHeight; fh++) {
                     int32_t inputH = oh * strideH - padH + fh;
                     if(inputH < 0) {
                       continue;
                     }
                     if(inputH >= inputHeight) {
                        break;
                     }

                     for(int32_t fw = 0; fw < filterWidth; fw++) {
                        int32_t inputW = ow * strideW - padW + fw;
                        if(inputW < 0) {
                          continue;
                        }
                        if(inputW >= inputWidth) {
                           break;
                        }

                        for(int32_t fd = 0; fd < filterDepth; fd++) {
                            int32_t inOffset = (inputH * inputWidth + inputW) * inputDepth + fd + g * filterDepth;
                            int32_t fOffset = (fh * filterWidth + fw) * filterDepth * outputDepth + fd * outputDepth;
                            sum += in[inOffset] * filter[fOffset + offset];
                        }//fd
                     }//fw
                  }// end of loop fh
                  sum += bias[offset];
                  sum = std::max(std::min(sum, outputActivationMax), outputActivationMin);
                  out[d] = sum;
              }// d
              out += outputGroupDepth;
          }//g
       }// end of loop ox
    }// end of loop oy
    return QNN_SUCCESS;
}

Qnn_ErrorHandle_t finalize(const CustomOp* operation) {
  QNN_CUSTOM_BE_ENSURE_EQ(operation->numInput(), 3, QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE)
  QNN_CUSTOM_BE_ENSURE_EQ(operation->numOutput(), 1, QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE)

  /**
   * Add code here
   **/

  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t free(CustomOp& operation) {

    /**
    * Add code here
    **/

    return QNN_SUCCESS;
}

Qnn_ErrorHandle_t populateFromNode(const QnnOpPackage_Node_t node,
                                   QnnOpPackage_GraphInfrastructure_t graphInfrastructure,
                                   CustomOp* operation) {
  // Add input
  for (uint32_t i = 0; i < numInputs(node); i++) {
    operation->addInput(getInput(node, i));
  }

  // Add output
  for (uint32_t i = 0; i < numOutputs(node); i++) {
    operation->addOutput(getOutput(node, i));
  }

  // Add params
   // The getParam function returns a pair -> hasParam, paramValue
   // Check that parameter has be retrieved. Pair.first is false if it was not found and the paramValue is nullptr

   auto groupPair = getParam(node, "group");

   QNN_CUSTOM_BE_ENSURE(groupPair.first, QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT)
   operation->addParam("group", groupPair.second);


   auto padsPair = getParam(node, "pads");

   QNN_CUSTOM_BE_ENSURE(padsPair.first, QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT)
   operation->addParam("pads", padsPair.second);


   auto stridesPair = getParam(node, "strides");

   QNN_CUSTOM_BE_ENSURE(stridesPair.first, QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT)
   operation->addParam("strides", stridesPair.second);


   auto dilationsPair = getParam(node, "dilations");

   QNN_CUSTOM_BE_ENSURE(dilationsPair.first, QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT)
   operation->addParam("dilations", dilationsPair.second);


   auto kernel_shapePair = getParam(node, "kernel_shape");

   QNN_CUSTOM_BE_ENSURE(kernel_shapePair.first, QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT)
   operation->addParam("kernel_shape", kernel_shapePair.second);


  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t validateOpConfig(Qnn_OpConfig_t opConfig) {
  QNN_CUSTOM_BE_ENSURE_EQ(
      strcmp(opConfig.v1.typeName, "Conv"), 0, QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT)

  QNN_CUSTOM_BE_ENSURE_EQ(opConfig.v1.numOfInputs, 3, QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE)
  QNN_CUSTOM_BE_ENSURE_EQ(opConfig.v1.numOfOutputs, 1, QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE)

  return QNN_SUCCESS;
}
}  // namespace conv

CustomOpRegistration_t* register_ConvCustomOp() {
  using namespace conv;
  static CustomOpRegistration_t ConvRegister = {execute, finalize, free, validateOpConfig, populateFromNode};
  return &ConvRegister;
}

REGISTER_OP(Conv, register_ConvCustomOp);