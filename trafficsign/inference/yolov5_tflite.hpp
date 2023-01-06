#pragma once

#include <cstdint>
#include <vector>
#include <chrono>
#include <iostream>
#include <fstream>
#include <cmath>
#include <glib.h>

#include <tensorflow/lite/model.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/delegates/gpu/cl/gpu_api_delegate.h>
#include <tensorflow/lite/delegates/gpu/common/model_builder.h>
#include <tensorflow/lite/delegates/gpu/common/status.h>
#include <tensorflow/lite/delegates/gpu/delegate.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>

#include "json/json.h"

struct Prediction
{
    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> labels;
};

class BBox {
 public:
  BBox() = default;
  ~BBox() = default;

  BBox(float x1, float y1, float x2, float y2) {
    xmin_ = x1;
    ymin_ = y1;
    xmax_ = x2;
    ymax_ = y2;
  }

 public:
  float xmin_;
  float ymin_;
  float xmax_;
  float ymax_;
};

class Anchor {
 public:
  Anchor() = default;
  ~Anchor() = default;
  bool operator<(const Anchor &t) const { return score_ < t.score_; }
  bool operator>(const Anchor &t) const { return score_ > t.score_; }

 public:
  float score_;
  int class_index;                   // cls score
  BBox finalbox_;        // final box res
};

class YOLOV5
{
public:
    YOLOV5() {}
    YOLOV5(const std::string json_file) {

        Json::Value root;
        std::ifstream ifs;
        ifs.open(json_file);

        Json::CharReaderBuilder builder;
        builder["collectComments"] = true;
        JSONCPP_STRING errs;
        if (!parseFromStream(builder, ifs, &root, &errs)) {
            std::cout << errs << std::endl;
        }
        std::cout << "parse json file " << json_file << " successfully in "<< __FUNCTION__ << std::endl;

        int size = root.size();   // count of root node
        for (int index = 0; index < size; index++) {
        
            const Json::Value inference_obj = root["inference"];
            int size = inference_obj.size();

            for(int index = 0; index < size; index++) {
                if (inference_obj[index].isMember("confThreshold")) {
                    this->conf_thres = inference_obj[index]["confThreshold"].asFloat();
                }
                if (inference_obj[index].isMember("nmsThreshold")) {
                    this->iou_thre = inference_obj[index]["nmsThreshold"].asFloat();
                }
                if (inference_obj[index].isMember("results")) {
                    this->nthreads = inference_obj[index]["results"].asInt();
                }
                if (inference_obj[index].isMember("model_path")) {
                    this->loadModel(inference_obj[index]["model_path"].asString());
                }
                if (inference_obj[index].isMember("labels_path")) {
                    this->getLabelsName(inference_obj[index]["labels_path"].asString(),this->labelNames);
                }
            }
        }
        ifs.close();
    }
    // Take a model path as string
    void loadModel(const  std::string path);
    // Take an image and return a prediction
    void run(cv::Mat image, Prediction &out_pred, std::vector<Anchor> &filtered_outputs);
    int PreProcess(cv::Mat& output, cv::Mat input);
    int PostProcess(std::vector<Anchor>& filtered_outputs,std::vector<float*> network_outputs);

    void getLabelsName(std::string path, std::vector<std::string> &labelNames);

    // thresh hold
    float conf_thres = 0.4;
    float iou_thre = 0.1;

    //   
    int pad_left = 0;
    int pad_top = 0;
    float scale = 0.;

    // 320 anchor info
    int nc = 43;  //
    std::array<int, 2> model_shape {320, 320}; // 320 
    std::vector<int> grids {40,40,20,20,10,10}; // 40 40 20 20 10 10
    std::vector<int> anchors {5,6, 8,15, 16,11, 15,30, 31,22, 29,59, 58,45, 78,99, 186,163};

    // number of threads
    int nthreads = 4;

    std::vector<std::string> labelNames;

private:
    // model's
    std::unique_ptr<tflite::FlatBufferModel> _model;
    std::unique_ptr<tflite::Interpreter> _interpreter;

    // parameters of interpreter's input
    int _input;
    int _in_height;
    int _in_width;
    int _in_channels;
    int _in_type;

    // parameters of original image
    int _img_height;
    int _img_width;

    // Input of the interpreter
    uint8_t *_input_8;

    TfLiteDelegate *_delegate;

    template <typename T>
    void fill(T *in, cv::Mat &src);
    void preprocess(cv::Mat &image);
    std::vector<std::vector<float>> tensorToVector2D(TfLiteTensor *pOutputTensor, const int &row, const int &colum);
    std::vector<float*> tensorToFloatPtr2D(TfLiteTensor *pOutputTensor, const int &row, const int &colum);
    void nonMaximumSupprition(
        std::vector<std::vector<float>> &predV,
        const int &row,
        const int &colum,
        std::vector<cv::Rect> &boxes,
        std::vector<float> &confidences,
        std::vector<int> &classIds,
        std::vector<int> &indices);
};