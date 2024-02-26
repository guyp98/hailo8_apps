#ifndef _HAILO_HW_C_HPP_
#define _HAILO_HW_C_HPP_ 

#include "hailo_objects.hpp"
#include "hailo_tensors.hpp"
// #include <stdio.h>
// #include <stdlib.h>
#include <iostream>
// #include "hailo/hailort.h"

#define REQUIRE_ACTION(cond, action, label, ...)     \
    do                                               \
    {                                                \
        if (!(cond))                                 \
        {                                            \
            std::cout << (__VA_ARGS__) << std::endl; \
            action;                                  \
            goto label;                              \
        }                                            \
    } while (0)

#define REQUIRE_SUCCESS(status, label, ...) REQUIRE_ACTION((HAILO_SUCCESS == (status)), , label, __VA_ARGS__)

class hailo_hw_c
{
public:
    // Constructor
    hailo_hw_c(hailo_status& status);

    // Copy constructor
    hailo_hw_c(const hailo_hw_c& other) = delete;
    hailo_hw_c& operator=(const hailo_hw_c& other) = delete;

    // Destructor
    ~hailo_hw_c() {
        (void)hailo_release_output_vstreams(output_vstreams, OUTPUT_COUNT);
        (void)hailo_release_hef(hef);
        (void)hailo_release_vdevice(vdevice);
    }

    inline hailo_input_vstream* get_input_vstreams() {
        return input_vstreams;
    }
    
    inline hailo_output_vstream* get_output_vstreams() {
        return output_vstreams;
    }

    inline int get_input_vstreams_size() {
        return INPUT_COUNT;
    }

    inline int get_output_vstreams_size() {
        return OUTPUT_COUNT;
    }

private:
    hailo_input_vstream input_vstreams[INPUT_COUNT] = {NULL};
    hailo_output_vstream output_vstreams[OUTPUT_COUNT] = {NULL};
    hailo_vdevice vdevice = NULL;
    hailo_hef hef = NULL;
    bool quantized = true;
};

#endif // _HAILO_HW_C_HPP_

