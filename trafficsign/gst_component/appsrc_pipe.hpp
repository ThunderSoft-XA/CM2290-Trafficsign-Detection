#ifndef APPSRC_PIPE_HPP
#define APPSRC_PIPE_HPP

#include <stdlib.h>

#include <gst/gst.h>
#include <gst/app/gstappsrc.h>

#include <opencv2/opencv.hpp>

#include "json/json.h"
#include "easy_queue.hpp"

class APPSrc2UdpSink {

private: 
	GstElement *pipeline, *appsrc, *videoconvert, *convert_caps, *qtic2venc, *h264parse, *rtph264pay,*mpegtsmux, *udpsink;
	GstAppSrcCallbacks cbs;
	/* Pointer to hold bus address */
	GstBus *bus = nullptr;

	std::string pipe_name;
	int width, height, framerate;
	std::string host = "127.0.0.1";
	int port = 5000;
public:
	GMainLoop *loop;
	guint sourceid;        /* To control the GSource */

	APPSrc2UdpSink(std::string json_file);
	int initPipe();
	Queue<cv::Mat> push_mat_queue;

	int checkElements();

	void setProperty();
	void updateCaps(int width, int height);


	int runPipe();
	~APPSrc2UdpSink();

};

#endif //APPSRC_PIPE_HPP