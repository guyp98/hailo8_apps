#ifndef _HAILO_HW_C_HPP_
#define _HAILO_HW_C_HPP_ 

#include "hailo_objects.hpp"
#include "hailo_tensors.hpp"

class hailo_hw_c
{
public:
    // Constructor
    hailo_hw_c(hailo_status& status) {
        hailo_configure_params_t config_params = {0};
        hailo_configured_network_group network_group = NULL;
        size_t network_group_size = 1;
        hailo_input_vstream_params_by_name_t input_vstream_params[INPUT_COUNT] = {0};
        hailo_output_vstream_params_by_name_t output_vstream_params[OUTPUT_COUNT] = {0};
        size_t input_vstreams_size = INPUT_COUNT;
        size_t output_vstreams_size = OUTPUT_COUNT;

        status = hailo_create_vdevice(NULL, &vdevice);
        REQUIRE_SUCCESS(status, l_exit, "Failed to create vdevice");

        status = hailo_create_hef_file(&hef, HEF_FILE);
        REQUIRE_SUCCESS(status, l_release_vdevice, "Failed reading hef file");

        status = hailo_init_configure_params_by_vdevice(hef, vdevice, &config_params);
        REQUIRE_SUCCESS(status, l_release_hef, "Failed initializing configure parameters");

        status = hailo_configure_vdevice(vdevice, hef, &config_params, &network_group, &network_group_size);
        REQUIRE_SUCCESS(status, l_release_hef, "Failed configure vdevice from hef");
        REQUIRE_ACTION(network_group_size == 1, status = HAILO_INVALID_ARGUMENT, l_release_hef, 
            "Invalid network group size");


        // Set input format type to auto, and mark the data as quantized - libhailort will not scale the data before writing to the HW
        quantized = true;
        status = hailo_make_input_vstream_params(network_group, quantized, HAILO_FORMAT_TYPE_AUTO,
            input_vstream_params, &input_vstreams_size);
        REQUIRE_SUCCESS(status, l_release_hef, "Failed making input virtual stream params");

        /* The input format order in the example HEF is NHWC in the user-side (may be seen using 'hailortcli parse-hef <HEF_PATH>).
        Here we override the user-side format order to be NCHW */
        for (size_t i = 0 ; i < input_vstreams_size; i++) {
            input_vstream_params[i].params.user_buffer_format.order = HAILO_FORMAT_ORDER_AUTO;
        }

        // Set output format type to float32, and mark the data as not quantized - libhailort will de-quantize the data after reading from the HW
        // Note: this process might affect the overall performance
        quantized = true;
        status = hailo_make_output_vstream_params(network_group, quantized, HAILO_FORMAT_TYPE_AUTO,
            output_vstream_params, &output_vstreams_size);
        REQUIRE_SUCCESS(status, l_release_hef, "Failed making output virtual stream params");

        REQUIRE_ACTION((input_vstreams_size <= INPUT_COUNT || output_vstreams_size <= OUTPUT_COUNT),
            status = HAILO_INVALID_OPERATION, l_release_hef, "Trying to infer network with too many input/output virtual "
            "streams, (either change HEF or change the definition of INPUT_COUNT, OUTPUT_COUNT)\n");

        status = hailo_create_input_vstreams(network_group, input_vstream_params, input_vstreams_size, input_vstreams);
        REQUIRE_SUCCESS(status, l_release_hef, "Failed creating virtual input streams\n");

        status = hailo_create_output_vstreams(network_group, output_vstream_params, output_vstreams_size, output_vstreams);
        REQUIRE_SUCCESS(status, l_release_input_vstream, "Failed creating output virtual streams\n");


        std::printf("build successfully\n");
        status = HAILO_SUCCESS;
        return;
    // l_release_output_vstream:
    //     (void)hailo_release_output_vstreams(output_vstreams, output_vstreams_size);
    l_release_input_vstream:
        (void)hailo_release_input_vstreams(input_vstreams, input_vstreams_size);
    l_release_hef:
        (void) hailo_release_hef(hef);
    l_release_vdevice:
        (void) hailo_release_vdevice(vdevice);
    l_exit:
        return;
    }

    // Copy constructor
    hailo_hw_c(const hailo_hw_c& other) = delete;
    hailo_hw_c& operator=(const hailo_hw_c& other) = delete;

    // Destructor
    ~hailo_hw_c() {
        (void)hailo_release_output_vstreams(output_vstreams, OUTPUT_COUNT);
        (void) hailo_release_hef(hef);
        (void) hailo_release_vdevice(vdevice);
    }

    hailo_input_vstream* get_input_vstreams() {
        return input_vstreams;
    }
    
    hailo_output_vstream* get_output_vstreams() {
        return output_vstreams;
    }

    int get_input_vstreams_size() {
        return INPUT_COUNT;
    }

    int get_output_vstreams_size() {
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

