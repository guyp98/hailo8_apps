/**
 * Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
 **/
#include <cmath>
#include <vector>
#include <algorithm>
#include "yolo_output.hpp"

std::pair<uint, float> YoloOutputLayer::get_class(uint row, uint col, uint anchor)
{
    uint cls_prob, prob_max = 0;
    uint selected_class_id = 1;
    for (uint class_id = label_offset; class_id <= _num_classes; class_id++)
    {
        cls_prob = get_class_prob(row, col, anchor, class_id);
        if (cls_prob > prob_max)
        {
            selected_class_id = class_id;
            prob_max = cls_prob;
        }
    }
    return std::pair<uint, float>(selected_class_id, get_class_conf(prob_max));
}

float YoloOutputLayer::get_confidence(uint row, uint col, uint anchor)
{
    uint channel = _tensor->features() / NUM_ANCHORS * anchor + CONF_CHANNEL_OFFSET;
    float confidence = _tensor->get_full_percision(row, col, channel, _is_uint16);
    if (_perform_sigmoid)
        confidence = sigmoid(confidence);
    return confidence;
}

float YoloOutputLayer::sigmoid(float x)
{
    // returns the value of the sigmoid function f(x) = 1/(1 + e^-x)
    return 1.0f / (1.0f + expf(-x));
}

uint YoloOutputLayer::get_class_prob(uint row, uint col, uint anchor, uint class_id)
{
    uint channel = _tensor->features() / NUM_ANCHORS * anchor + CLASS_CHANNEL_OFFSET + class_id - 1;
    if (_is_uint16)
        return _tensor->get_uint16(row, col, channel);
    else
        return _tensor->get(row, col, channel);
}

float Yolov5OL::get_class_conf(uint prob_max)
{
    float conf = _tensor->fix_scale(prob_max);
    if (_perform_sigmoid)
        conf = sigmoid(conf);
    return conf;
}

std::pair<float, float> Yolov5OL::get_center(uint row, uint col, uint anchor)
{
    float x, y = 0.0f;
    uint channel = _tensor->features() / NUM_ANCHORS * anchor;
    x = (_tensor->get_full_percision(row, col, channel, _is_uint16) * 2.0f - 0.5f + col) / _width;
    y = (_tensor->get_full_percision(row, col, channel + 1, _is_uint16) * 2.0f - 0.5f + row) / _height;
    return std::pair<float, float>(x, y);
}

std::pair<float, float> Yolov5OL::get_shape(uint row, uint col, uint anchor, uint image_width, uint image_height)
{
    float w, h = 0.0f;
    uint channel = _tensor->features() / NUM_ANCHORS * anchor + NUM_CENTERS;
    w = pow(2.0f * _tensor->get_full_percision(row, col, channel, _is_uint16), 2.0f) * _anchors[anchor * 2] / image_width;
    h = pow(2.0f * _tensor->get_full_percision(row, col, channel + 1, _is_uint16), 2.0f) * _anchors[anchor * 2 + 1] / image_height;
    return std::pair<float, float>(w, h);
}

float Yolov3OL::get_class_conf(uint prob_max)
{
    float conf = _tensor->fix_scale(prob_max);
    if (_perform_sigmoid)
        conf = sigmoid(conf);
    return conf;
}

std::pair<float, float> Yolov3OL::get_shape(uint row, uint col, uint anchor, uint image_width, uint image_height)
{
    float w, h = 0.0f;
    uint channel = _tensor->features() / NUM_ANCHORS * anchor + NUM_CENTERS;
    w = expf(_tensor->get_full_percision(row, col, channel, _is_uint16)) * _anchors[anchor * 2] / image_width;
    h = expf(_tensor->get_full_percision(row, col, channel + 1, _is_uint16)) * _anchors[anchor * 2 + 1] / image_height;
    return std::pair<float, float>(w, h);
}

std::pair<float, float> Yolov3OL::get_center(uint row, uint col, uint anchor)
{
    float x, y = 0.0f;
    uint channel = _tensor->features() / NUM_ANCHORS * anchor;
    x = (sigmoid(_tensor->get_full_percision(row, col, channel, _is_uint16)) + col) / _width;
    y = (sigmoid(_tensor->get_full_percision(row, col, channel + 1, _is_uint16)) + row) / _height;
    return std::pair<float, float>(x, y);
}

float Yolov4OL::get_confidence(uint row, uint col, uint anchor)
{
    float confidence = _obj->get_full_percision(row, col, anchor, _is_uint16);
    if (_perform_sigmoid)
        confidence = sigmoid(confidence);
    return confidence;
}

uint Yolov4OL::get_class_prob(uint row, uint col, uint anchor, uint class_id)
{
    uint channel = _num_classes * anchor + class_id - 1;
    return _cls->get(row, col, channel);
}

float Yolov4OL::get_class_conf(uint prob_max)
{
    float class_conf = _cls->fix_scale(prob_max);
    if (_perform_sigmoid)
        class_conf = sigmoid(class_conf);
    return class_conf;
}

std::pair<float, float> Yolov4OL::get_center(uint row, uint col, uint anchor)
{
    uint channel = (_center->features() / NUM_ANCHORS) * anchor;
    float x = (_center->get_full_percision(row, col, channel, _is_uint16) * SCALE_XY - 0.5f * (SCALE_XY - 1) + col) / _width;
    float y = (_center->get_full_percision(row, col, channel + 1, _is_uint16) * SCALE_XY - 0.5f * (SCALE_XY - 1) + row) / _height;
    return std::pair<float, float>(x, y);
}

std::pair<float, float> Yolov4OL::get_shape(uint row, uint col, uint anchor, uint image_width, uint image_height)
{
    float w, h = 0.0f;
    uint channel = (_scale->features() / NUM_ANCHORS) * anchor;
    w = expf(_scale->get_full_percision(row, col, channel, _is_uint16)) * _anchors[anchor * 2] / image_width;
    h = expf(_scale->get_full_percision(row, col, channel + 1, _is_uint16)) * _anchors[anchor * 2 + 1] / image_height;
    return std::pair<float, float>(w, h);
}

std::pair<float, float> TinyYolov4OL::get_center(uint row, uint col, uint anchor)
{
    uint channel = (_tensor->features() / NUM_ANCHORS) * anchor;
    float x = (sigmoid(_tensor->get_full_percision(row, col, channel, _is_uint16)) * SCALE_XY - 0.5f * (SCALE_XY - 1) + col) / _width;
    float y = (sigmoid(_tensor->get_full_percision(row, col, channel + 1, _is_uint16)) * SCALE_XY - 0.5f * (SCALE_XY - 1) + row) / _height;
    return std::pair<float, float>(x, y);
}

float TinyYolov4OL::get_class_conf(uint prob_max)
{
    float conf = _tensor->fix_scale(prob_max);
    if (_perform_sigmoid)
        conf = sigmoid(conf);
    return conf;
}

std::pair<float, float> TinyYolov4OL::get_shape(uint row, uint col, uint anchor, uint image_width, uint image_height)
{
    float w, h = 0.0f;
    uint channel = _tensor->features() / NUM_ANCHORS * anchor + NUM_CENTERS;
    w = expf(_tensor->get_full_percision(row, col, channel, _is_uint16)) * _anchors[anchor * 2] / image_width;
    h = expf(_tensor->get_full_percision(row, col, channel + 1, _is_uint16)) * _anchors[anchor * 2 + 1] / image_height;
    return std::pair<float, float>(w, h);
}

float YoloXOL::get_confidence(uint row, uint col, uint anchor)
{
    float confidence = _obj->get_full_percision(row, col, 0, _is_uint16);
    if (_perform_sigmoid)
        confidence = sigmoid(confidence);
    return confidence;
}

uint YoloXOL::get_class_prob(uint row, uint col, uint anchor, uint class_id)
{
    return _cls->get(row, col, class_id - 1);
}

float YoloXOL::get_class_conf(uint prob_max)
{
    float conf = _cls->fix_scale(prob_max);
    if (_perform_sigmoid)
        conf = sigmoid(conf);
    return conf;
}

std::pair<float, float> YoloXOL::get_center(uint row, uint col, uint anchor)
{
    float x, y = 0.0f;
    x = (_bbox->get_full_percision(row, col, 0, _is_uint16) + col) / _width;
    y = (_bbox->get_full_percision(row, col, 1, _is_uint16) + row) / _height;
    return std::pair<float, float>(x, y);
}

std::pair<float, float> YoloXOL::get_shape(uint row, uint col, uint anchor, uint image_width, uint image_height)
{
    float w, h = 0.0f;
    w = expf(_bbox->get_full_percision(row, col, 2, _is_uint16)) / _width;
    h = expf(_bbox->get_full_percision(row, col, 3, _is_uint16)) / _height;
    return std::pair<float, float>(w, h);
}
