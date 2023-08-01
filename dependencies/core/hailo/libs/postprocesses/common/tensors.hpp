/**
* Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
* Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
**/
#pragma once

#include "hailo_objects.hpp"
#include "xtensor/xadapt.hpp"
#include "xtensor/xarray.hpp"

namespace common
{
    //-------------------------------
    // COMMON TRANSFORMS
    //-------------------------------
    // template <typename T>
    // xt::xarray<float> dequantize(const xt::xarray<T> &input, const float &qp_scale, const float &qp_zp);
    
    xt::xarray<float> dequantize(const xt::xarray<uint8_t> &input, const float &qp_scale, const float &qp_zp);
    xt::xarray<float> dequantize(const xt::xarray<uint16_t> &input, const float &qp_scale, const float &qp_zp);
    xt::xarray<float> dequantize(const xt::xarray<float> &input, const float &qp_scale, const float &qp_zp);

    xt::xarray<uint8_t> get_xtensor(HailoTensorPtr &tensor);

    xt::xarray<uint16_t> get_xtensor_uint16(HailoTensorPtr &tensor);

    xt::xarray<float> get_xtensor_float(HailoTensorPtr &tensor);
    
    std::vector<HailoTensorPtr> get_tensor_values(const std::map<std::string, HailoTensorPtr> &tensors);

}