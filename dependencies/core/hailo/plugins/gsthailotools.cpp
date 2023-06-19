/**
* Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
* Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
**/
#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <gst/gst.h>
#include "filter/gsthailofilter.hpp"
#include "filter/gsthailocounter.hpp"
#include "muxer/gsthailomuxer.hpp"
#include "muxer/gsthailoroundrobin.hpp"
#include "muxer/gsthailostreamrouter.hpp"
#include "cropping/gsthailoaggregator.hpp"
#include "cropping/gsthailocropper.hpp"
#include "overlay/gsthailooverlay.hpp"
#include "gst_hailo_meta.hpp"
#include "gst_hailo_cropping_meta.hpp"
#include "gst_hailo_stream_meta.hpp"
#include "tiling/gsthailotilecropper.hpp"
#include "tiling/gsthailotileaggregator.hpp"
#include "tracking/gsthailotracker.hpp"
#include "gallery/gsthailogallery.hpp"
#include "export/export_file/gsthailoexportfile.hpp"
#include "export/export_zmq/gsthailoexportzmq.hpp"
#include "import/import_zmq/gsthailoimportzmq.hpp"


static gboolean
plugin_init(GstPlugin *plugin)
{
    gst_element_register(plugin, "hailooverlay", GST_RANK_PRIMARY, GST_TYPE_HAILO_OVERLAY);
    gst_element_register(plugin, "hailofilter", GST_RANK_PRIMARY, GST_TYPE_HAILO_FILTER);
    gst_element_register(plugin, "hailocounter", GST_RANK_PRIMARY, GST_TYPE_HAILO_COUNTER);
    gst_element_register(plugin, "hailomuxer", GST_RANK_PRIMARY, GST_TYPE_HAILO_MUXER);
    gst_element_register(plugin, "hailoroundrobin", GST_RANK_PRIMARY, GST_TYPE_HAILO_ROUND_ROBIN);
    gst_element_register(plugin, "hailostreamrouter", GST_RANK_PRIMARY, GST_TYPE_HAILO_STREAM_ROUTER);
    gst_element_register(plugin, "hailocropper", GST_RANK_PRIMARY, GST_TYPE_HAILO_CROPPER);
    gst_element_register(plugin, "hailotilecropper", GST_RANK_PRIMARY, GST_TYPE_HAILO_TILE_CROPPER);
    gst_element_register(plugin, "hailoaggregator", GST_RANK_PRIMARY, GST_TYPE_HAILO_AGGREGATOR);
    gst_element_register(plugin, "hailotileaggregator", GST_RANK_PRIMARY, GST_TYPE_HAILO_TILE_AGGREGATOR);
    gst_element_register(plugin, "hailotracker", GST_RANK_PRIMARY, GST_TYPE_HAILO_TRACKER);
    gst_element_register(plugin, "hailogallery", GST_RANK_PRIMARY, GST_TYPE_HAILO_GALLERY);
    gst_element_register(plugin, "hailoexportfile", GST_RANK_PRIMARY, GST_TYPE_HAILO_EXPORT_FILE);
    gst_element_register(plugin, "hailoexportzmq", GST_RANK_PRIMARY, GST_TYPE_HAILO_EXPORT_ZMQ);
    gst_element_register(plugin, "hailoimportzmq", GST_RANK_PRIMARY, GST_TYPE_HAILO_IMPORT_ZMQ);
    gst_hailo_meta_get_info();
    gst_hailo_meta_api_get_type();
    gst_hailo_cropping_meta_get_info();
    gst_hailo_cropping_meta_api_get_type();
    gst_hailo_stream_meta_get_info();
    gst_hailo_stream_meta_api_get_type();

    return TRUE;
}

GST_PLUGIN_DEFINE(GST_VERSION_MAJOR, GST_VERSION_MINOR, hailotools, "hailo tools plugin", plugin_init,
                  VERSION, "unknown", PACKAGE, "https://hailo.ai/")
