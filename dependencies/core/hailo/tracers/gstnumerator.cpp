/* GstShark - A Front End for GstTracer
 * Copyright (C) 2016-2018 RidgeRun Engineering <carlos.rodriguez@ridgerun.com>
 *
 * This file is part of GstShark.
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 */
/**
 * SECTION:gstnumerator
 * @short_description: log current idendity
 *
 */

#include "gstnumerator.hpp"
#include "gstctf.hpp"
#include <stdio.h>
#include <stdbool.h>
#include <map>

GST_DEBUG_CATEGORY_STATIC(gst_numerator_debug);
#define GST_CAT_DEFAULT gst_numerator_debug
#define MAX_STREAMS 256

struct _GstNumeratorTracer
{
    GstSharkTracer parent;
};

#define _do_init \
    GST_DEBUG_CATEGORY_INIT(gst_numerator_debug, "numerator", 0, "numerator tracer");

G_DEFINE_TYPE_WITH_CODE(GstNumeratorTracer, gst_numerator_tracer,
                        GST_SHARK_TYPE_TRACER, _do_init);

static GstElementFactory *avdec_factory = NULL;

static GstElement *
get_parent_element(GstPad *pad)
{
    GstElement *element;
    GstObject *parent;
    GstObject *child = GST_OBJECT(pad);

    do
    {
        parent = GST_OBJECT_PARENT(child);

        if (GST_IS_ELEMENT(parent))
            break;

        child = parent;

    } while (GST_IS_OBJECT(child));

    element = gst_pad_get_parent_element(GST_PAD(child));

    return element;
}

static gboolean is_decoder(GstElement *element);

static gboolean is_decoder(GstElement *element)
{
    GstElementFactory *efactory;
    g_return_val_if_fail(element, FALSE);
    efactory = gst_element_get_factory(element);
    return (efactory == avdec_factory);
}

static void gst_numerator_buffer_pre(GObject *self, GstClockTime ts,
                                     GstPad *pad, GstBuffer *buffer);

std::map<gchar *, int> map_stream_ids_offsets; // stream_id, offset
static void gst_numerator_buffer_pre(GObject *self, GstClockTime ts, GstPad *pad, GstBuffer *buffer)
{
    GstElement *element;
    gchar *stream_id;
    element = get_parent_element(pad);

    if (!is_decoder(element))
    {
        return;
    }

    stream_id = gst_pad_get_stream_id(pad);

    auto pair = map_stream_ids_offsets.find(stream_id);
    if (pair != map_stream_ids_offsets.end())
    {
        pair->second++; // increment offset
        buffer->offset = pair->second;
    }
    else
    {
        map_stream_ids_offsets.insert(std::pair<gchar *, int>(stream_id, 1));
        buffer->offset = 1;
    }

    g_free(stream_id);
    gst_object_unref(element);
}

/* tracer class */
static void
gst_numerator_tracer_class_init(GstNumeratorTracerClass *klass) {}

static void
gst_numerator_tracer_init(GstNumeratorTracer *self)
{
    GstSharkTracer *tracer = GST_SHARK_TRACER(self);

    /* Find the decoder factory that is going to be compared against
    the element under inspection to see if it is a decoder */
    avdec_factory = gst_element_factory_find("avdec_h264");

    gst_shark_tracer_register_hook(tracer, "pad-push-pre",
                                   G_CALLBACK(gst_numerator_buffer_pre));
}
