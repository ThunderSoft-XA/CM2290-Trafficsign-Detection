#include "yolov5_tflite.hpp"

int main(int argc, char *argv[])
{
    if (argc != 5)
    {
        std::cout << "\nError! Usage: <path to tflite model> <path to classes names> <path to input img> <input size>\n"
                  << std::endl;
        return 1;
    }

    Prediction out_pred;
    const std::string model_path = argv[1];
    const std::string names_path = argv[2];
    const std::string img_path = argv[3];
    int input_size = atoi(argv[4]);
    // const std::string save_path = argv[4];

    std::vector<std::string> labelNames;

    YOLOV5 model;

    // conf
    model.conf_thres = 0.30;
    model.iou_thre = 0.40;
    model.nthreads = 4;

    // Load the saved_model
    model.loadModel(model_path);

    // Load names
    model.getLabelsName(names_path, labelNames);

    std::cout << "\nLabel Count: " << labelNames.size() << "\n"
              << std::endl;

    // cv::VideoCapture capture;
    // if (all_of(video_path.begin(), video_path.end(), ::isdigit) == false)
    //     capture.open(video_path); 
    // else
    //     capture.open(stoi(video_path));

    // cv::Mat frame;
    // if (!capture.isOpened())
    //     throw "\nError when reading video steam\n";
    // cv::namedWindow("w", 1);

    // save video config
    bool save = true;
    // int frame_width = 320;
    // int frame_height = 320;
    // auto fourcc = capture.get(cv::CAP_PROP_FOURCC);
    // int frame_width = capture.get(cv::CAP_PROP_FRAME_WIDTH);
    // int frame_height = capture.get(cv::CAP_PROP_FRAME_WIDTH);
    // cv::VideoWriter video(save_path, fourcc, 30, cv::Size(frame_width, frame_height), true);

    // for (;;)
    // {
    //     capture >> frame;
    cv::Mat frame;
    frame = cv::imread(img_path);
    cv::Mat base_frame = frame.clone();
    if (frame.empty())
        std::cout << "test img is empty" << std::endl;
    std::cout << "load test img finished" << std::endl;
    // start
    auto start = std::chrono::high_resolution_clock::now();
    // Predict on the input image
    std::vector<Anchor> filtered_outputs;
    model.run(frame, out_pred,filtered_outputs);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "\nModel run time 'milliseconds': " << duration.count() << "\n"
                << std::endl;

    // add the bbox to the image and save it
    auto boxes = out_pred.boxes;
    auto scores = out_pred.scores;
    auto labels = out_pred.labels;

    for (int i = 0; i < boxes.size(); i++)
    {
        auto box = boxes[i];
        auto score = scores[i];
        auto label = labels[i];
        std::cout << "box num : " << i << ", score = " << score << ", label = " << labelNames[label] << std::endl;
        cv::rectangle(frame, box, cv::Scalar(255, 0, 0), 2);
        cv::putText(frame, labelNames[label], cv::Point(box.x, box.y), cv::FONT_HERSHEY_COMPLEX, 1.0, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    }
    cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
    // cv::imshow("output.png", frame);
    out_pred = {};
    if (save == true)
    {
        // cv::resize(frame, frame, cv::Size(input_size, input_size), 0, 0, true);
        // video.write(frame);
        cv::imwrite("/data/yolov5/out.png",frame);
    }

    for(int i = 0 ;i < filtered_outputs.size(); i++){
        BBox bbox = filtered_outputs[i].finalbox_;
        cv::Rect rect(bbox.xmin_,
            bbox.ymin_,
            bbox.xmax_ - bbox.xmin_,
            bbox.ymax_ - bbox.ymin_);
        cv::rectangle(base_frame, rect, cv::Scalar(255),2);
    }
    cv::imwrite("/data/yolov5/out_anchor.png", base_frame);
    return 0;
        // cv::waitKey(10);
    // }
    // capture.release();

    // if (save == true)
    // {
    //     video.release();
    // }
    // cv::destroyAllWindows();
}