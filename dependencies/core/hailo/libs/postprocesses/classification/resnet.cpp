/**
* Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
* Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
**/
#include <vector>
#include <iostream>
#include "common/labels/imagenet.hpp"
#include "common/tensors.hpp"
#include "common/math.hpp"
#include "resnet.hpp"
#include "xtensor/xadapt.hpp"
#include "xtensor/xarray.hpp"

#define OUTPUT_LAYER_NAME "resnet_v1_50/softmax1"
#define COMMA ","

void top1(HailoROIPtr roi)
{
    const int k = 1;
    std::string label = "";

    if (!roi->has_tensors())
    {
        return;
    }

    // Extract the relevant output tensor.
    HailoTensorPtr scores = roi->get_tensor(OUTPUT_LAYER_NAME);

    // Convert the tensor to xarray.
    xt::xarray<uint8_t> xscores = common::get_xtensor(scores);

    // Find the topk scores.
    xt::xarray<int> top_k_scores = common::top_k(xscores, k);

    // Extrats the label of the top score.
    int index = top_k_scores[0];
    std::string labels = common::imagenet_labels[index];

    // If there are multiple synonyms for this class, take only the first.
    int comma_pos = labels.find(COMMA);
    if (comma_pos > 0)
        label = labels.substr(0, comma_pos);
    else
        label = labels;
    float confidence = scores->fix_scale(xscores(index));
    // Update the tensor with the classification result.
    hailo_common::add_classification(roi,
                                     std::string("imagenet"),
                                     label,
                                     confidence,
                                     index);
}

void filter(HailoROIPtr roi)
{
    top1(roi);
}
