#ifndef CAMERA_PIPE_HPP
#define CAMERA_PIPE_HPP

#include <iostream>
#include <fstream>
#include <string>

#include <gst/gst.h>
#include <gst/audio/audio.h>

#include <opencv2/opencv.hpp>

#include "json/json.h"
#include "easy_queue.hpp"

#define CHUNK_SIZE 1024   /* Amount of bytes we are sending in each buffer */
#define SAMPLE_RATE 44100 /* Samples per second we are sending */

/* Structure to contain all our information, so we can pass it to callbacks */
class CameraPipe {

private:
	GstElement *pipeline;
	GstElement *qtiqmmf, *qmmf_caps, *qtic2venc, *h264parse;
	GstElement *qtivdec, *videoconvert, *convert_caps, *app_sink;
	GstBus *bus;

	std::string pipe_name;
	int camera_id = 0;
    int width = 640, height = 480, framerate = 30;

	guint64 num_samples;   /* Number of samples generated so far (for timestamp generation) */
	gfloat a, b, c, d;     /* For waveform generation */

	guint sourceid;        /* To control the GSource */

public:
	CameraPipe(std::string json_file);
	int initPipe();

	int checkElements();

	void setProperty();

	int runPipeline();

	~CameraPipe() {
		/* Free resources */
		gst_element_set_state (this->pipeline, GST_STATE_NULL);
		gst_object_unref (this->pipeline);
	}
};

#endif  // CAMERA_PIPE_HPP