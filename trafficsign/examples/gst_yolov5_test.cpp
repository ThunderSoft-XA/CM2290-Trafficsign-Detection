#include <stdio.h>
#include <unistd.h>
#include <thread>

#include "camera_pipe.hpp"
#include "appsrc_pipe.hpp"
#include "utils.hpp"

#include "yolov5_tflite.hpp"

extern Queue<cv::Mat> rgb_mat_queue;

int main(int argc, char **argv) 
{
	std::string json_file = "./gst_yolov5_config.json";
    // if( parse_arg(argc, argv, json_file) != 0) {
	// 	std::cout << "invalid argument, exit !!" << std::endl;
	// 	return -1;
	// }

    /* Initialize GStreamer */
    gst_init (nullptr, nullptr);
    GMainLoop *main_loop;  /* GLib's Main Loop */
    main_loop = g_main_loop_new (NULL, FALSE);

    // constructot required class
    Queue<cv::Mat> push_queue;
    CameraPipe camera_pipe(json_file);
    YOLOV5 model(json_file);
    APPSrc2UdpSink push_pipe(json_file);
    push_pipe.push_mat_queue = push_queue;

    // std::cout << "module init finished !!!" << std::endl;
    g_print("module init finished !!!\n");

    Prediction out_pred;

    if( camera_pipe.initPipe() == -1 || (push_pipe.initPipe() == -1) ) {
        std::cout << "init pipeline failed , exit" << std::endl;
        goto error;
    }
    std::cout << "init all pipeline finished" << std::endl;

    if( camera_pipe.checkElements() != 0 || push_pipe.checkElements() != 0) {
        std::cout << "member or element check failed , exit" << std::endl;
        goto error;
    }

    camera_pipe.setProperty();
    push_pipe.setProperty();
    std::cout << "all pipeline's property set finished" << std::endl;

    camera_pipe.runPipeline();
    std::cout << "runing camera pipeline .............."<< __FILE__ << "!!!" << std::endl;
    std::thread([&](){
        cv::Mat frame;
        // std::cout << "gst sample data for inference!!!" << std::endl;
        g_print("gst sample data for inference!!!\n");
        while( 1 ) {
            if(rgb_mat_queue.empty()) {
                continue;
            }
            frame = rgb_mat_queue.pop();
            cv::Mat base_frame = frame.clone();

            if( frame.empty()) {
                continue;
            }

            // start
            g_print("gst rgb_mat_queue mat data, inferencing!!!\n");
            auto start = std::chrono::high_resolution_clock::now();
            // Predict on the input image
            std::vector<Anchor> filtered_outputs;
            model.run(frame, out_pred,filtered_outputs);
            // model.run(frame, out_pred);
            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
            // std::cout << "\nModel run time 'milliseconds': " << duration.count() << "\n" << std::endl;
            g_print("Model run time 'milliseconds': %ld\n",duration.count());

            // add the bbox to the image and save it
            auto boxes = out_pred.boxes;
            auto scores = out_pred.scores;
            auto labels = out_pred.labels;

            for (int i = 0; i < boxes.size(); i++)
            {
                auto box = boxes[i];
                auto score = scores[i];
                auto label = labels[i];
                cv::rectangle(frame, box, cv::Scalar(255, 0, 0), 2);
                cv::putText(frame, model.labelNames[label], cv::Point(box.x, box.y), cv::FONT_HERSHEY_COMPLEX, 1.0, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
            }
            cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);

            for(int i = 0 ;i < filtered_outputs.size(); i++){
                BBox bbox = filtered_outputs[i].finalbox_;
                cv::Rect rect(bbox.xmin_,
                    bbox.ymin_,
                    bbox.xmax_ - bbox.xmin_,
                    bbox.ymax_ - bbox.ymin_);
                cv::rectangle(base_frame, rect, cv::Scalar(255),2);
            }
            // cv::imwrite("/data/yolov5/out_anchor.png", base_frame);

            push_pipe.push_mat_queue.push_back(base_frame);

            // if(first_appsrc) {
            //     push_pipe.runPipe(frame);
            //     push_pipe.pushMatData(frame);
            //     first_appsrc = false;
            // } else {
            //     push_pipe.pushMatData(frame);
            // }
            // cv::imshow("output", frame);
            out_pred = {};
            // cv::waitKey(10);
        }
    }).detach();

    push_pipe.updateCaps(push_pipe.push_mat_queue.pop().cols, push_pipe.push_mat_queue.pop().rows);
    std::cout << " push_queue size = " << push_pipe.push_mat_queue.size() << std::endl;
	// std::thread([&](){
        sleep(2);
		push_pipe.runPipe();
    // }).detach();

    g_main_loop_run (main_loop);

error:
    camera_pipe.~CameraPipe();
    push_pipe.~APPSrc2UdpSink();
    g_main_loop_unref (main_loop);
    return 0;
    
}