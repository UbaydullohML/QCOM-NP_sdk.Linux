//==============================================================================
//
//  Copyright (c) 2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <iostream>
#include <vector>
#include <string>
#include <assert.h>
#include <unordered_map>
#include <cstring>
#include <cstdlib>

#include "LoadInputTensor.hpp"
#include "Util.hpp"

#include "SNPE/SNPE.h"
#include "SNPE/SNPEUtil.h"
#include "DlSystem/ITensor.h"
#include "DlSystem/StringList.h"
#include "DlSystem/TensorMap.h"
#include "DlSystem/TensorShape.h"


// Load all the required input tensors for the network
std::tuple<Snpe_TensorMap_Handle_t, bool> LoadInputTensorMap(Snpe_SNPE_Handle_t& snpeHandle,
                                                             std::vector<std::string>& fileLines,
                                                             Snpe_StringList_Handle_t& inputTensorNamesHandle,
                                                             std::vector<Snpe_ITensor_Handle_t>& inputs){
    Snpe_TensorMap_Handle_t dummyInputTensorMapHandle;
    Snpe_TensorMap_Handle_t inputTensorMapHandle = Snpe_TensorMap_Create();

    for(size_t i=0; i<fileLines.size(); i++) {
        std::string fileLine(fileLines[i]);
        std::vector<std::string> filePaths;
        split(filePaths, fileLine, ' ');
        for (size_t j = 0; j<Snpe_StringList_Size(inputTensorNamesHandle); j++) {
            std::string filePath(filePaths[j]);
            std::cout << "\t" << j + 1 << ") " << filePath << std::endl;
            const char* inputName = Snpe_StringList_At(inputTensorNamesHandle, j);
            std::vector<float> inputVec = loadFloatDataFile(filePath);
            auto inputShapeHandle = Snpe_SNPE_GetInputDimensions(snpeHandle, inputName);
            if(inputShapeHandle == nullptr) throw std::runtime_error("Failed to obtain input dimensions");
            inputs[j] = Snpe_Util_CreateITensor(inputShapeHandle);
            Snpe_TensorShape_Delete(inputShapeHandle);
            const size_t inputSize_at_j = Snpe_ITensor_GetSize(inputs[j]);
            if(inputVec.size() != inputSize_at_j){
                std::cerr << "Size of input does not match network. \n"
                          << "Expecting: " << inputSize_at_j << "\n"
                          << "Got: " << inputVec.size() << "\n";
                return std::make_tuple(dummyInputTensorMapHandle, false);
            }
            std::copy(inputVec.begin(), inputVec.end(), (float*)Snpe_ITensor_GetData(inputs[j]));
            Snpe_TensorMap_Add(inputTensorMapHandle, inputName, inputs[j]);
        }
    }
    std::cout << "Finished processing inputs for current inference \n";
    return std::make_tuple(inputTensorMapHandle, true);
}
