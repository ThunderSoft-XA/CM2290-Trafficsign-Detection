#include "yolov5_tflite.hpp"

void YOLOV5::getLabelsName(std::string path, std::vector<std::string> &labelNames)
{
    // Open the File
    std::ifstream in(path.c_str());
    // Check if object is valid
    if (!in)
        throw std::runtime_error("Can't open ");
    std::string str;
    // Read the next line from File until it reaches the end.
    while (std::getline(in, str))
    {
        // Line contains string of length > 0 then save it in vector
        if (str.size() > 0)
            labelNames.push_back(str);
    }
    // Close The File
    in.close();
}

void YOLOV5::loadModel(const  std::string path)
{
    _model = tflite::FlatBufferModel::BuildFromFile(path.c_str());
    if (!_model) {
        std::cout << "\nFailed to load the model.\n"
                  << std::endl;
        exit(1);
    } else {
        std::cout << "\nModel loaded successfully.\n";
    }
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder(*_model, resolver)(&_interpreter);

    TfLiteGpuDelegateOptionsV2 options = TfLiteGpuDelegateOptionsV2Default();
    options.is_precision_loss_allowed = true;
    options.inference_preference = TFLITE_GPU_INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER;
    options.experimental_flags |= TFLITE_GPU_EXPERIMENTAL_FLAGS_ENABLE_QUANT;
    options.inference_priority1 = TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY;
    options.inference_priority2 = TFLITE_GPU_INFERENCE_PRIORITY_AUTO;
    options.inference_priority3 = TFLITE_GPU_INFERENCE_PRIORITY_AUTO;
    options.max_delegated_partitions = 1;

    auto* delegate = TfLiteGpuDelegateV2Create(&options);
    auto status = _interpreter->ModifyGraphWithDelegate(delegate);
    // auto status = _interpreter->AllocateTensors();
    if (status != kTfLiteOk) {
        std::cout << "Failed to allocate the memory for tensors." << std::endl;
        exit(1);
    } else {
        std::cout << "\nMemory allocated for tensors.\n";
    }

    // input information
    _input = _interpreter->inputs()[0];
    TfLiteIntArray *dims = _interpreter->tensor(_input)->dims;
    _in_height = dims->data[1];
    _in_width = dims->data[2];
    _in_channels = dims->data[3];
    _in_type = _interpreter->tensor(_input)->type;
    std::cout << "input width = " << _in_height << ", input height = " << _in_width << ", channels = " << _in_channels << ", type = " << _in_type << std::endl;
    _input_8 = _interpreter->typed_tensor<uint8_t>(_input);

    _interpreter->SetNumThreads(nthreads);
}

void YOLOV5::preprocess(cv::Mat &image)
{
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    cv::resize(image, image, cv::Size(_in_height, _in_width), cv::INTER_CUBIC);
    image.convertTo(image, CV_8U);
}

int YOLOV5::PreProcess(cv::Mat& output, cv::Mat input)
{
  output = cv::Mat(input);
  int img_h = input.rows;
  int img_w = input.cols;

  float r = std::min(1.0 * this->model_shape[0] / img_w,
     1.0 * this->model_shape[1] / img_h);

  int new_unpad_w = int(img_w * r + 0.5);
  int new_unpad_h = int(img_h * r + 0.5);
  float dw = this->model_shape[0] - new_unpad_w;
  float dh = this->model_shape[1] - new_unpad_h;

  dw /= 2;
  dh /= 2;

  if(!(new_unpad_w == img_w && new_unpad_h == img_h )){
    cv::resize(output, output, cv::Size(new_unpad_w, new_unpad_h));
  }

  int top    = int(dh + 0.4);
  int bottom = int(dh + 0.6);
  int left   = int(dw + 0.4);
  int right  = int(dw + 0.6);

  cv::copyMakeBorder(output, output, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(0,0,0));
  pad_left = left;
  pad_top = top;
  scale = r;
  return 0;
}

template <typename T>
void YOLOV5::fill(T *in, cv::Mat &src)
{
    int n = 0, nc = src.channels(), ne = src.elemSize();
    g_print("src.channels = %d,src.rows = %d,src.cols = %d\n",nc,src.rows,src.cols);
    if(!src.isContinuous()) {
        src = src.clone();
    }
    for (int y = 0; y < src.rows; ++y)
        for (int x = 0; x < src.cols; ++x)
            for (int c = 0; c < nc; ++c)
                in[n++] = src.data[y * src.rows + x * src.cols + c];
}

std::vector<std::vector<float>> YOLOV5::tensorToVector2D(TfLiteTensor *pOutputTensor, const int &row, const int &colum)
{

    auto scale = pOutputTensor->params.scale;
    auto zero_point = pOutputTensor->params.zero_point;
    std::vector<std::vector<float>> v;
    for (int32_t i = 0; i < row; i++)
    {
        std::vector<float> _tem;
        for (int j = 0; j < colum; j++)
        {
            // float val_float = (((int32_t)pOutputTensor->data.uint8[i * colum + j]) - zero_point) * scale;
            float val_float = ((int32_t)pOutputTensor->data.uint8[i * colum + j]);
            _tem.push_back(val_float);
        }
        v.push_back(_tem);
    }
    return v;
}

std::vector<float*> YOLOV5::tensorToFloatPtr2D(TfLiteTensor *pOutputTensor, const int &row, const int &colum)
{
    std::vector<float *> v;
    std::cout << "output row = " << row << ", output colcum = " << colum << std::endl;
    std::cout << "pOutputTensor float size " << pOutputTensor->bytes << std::endl;
    int ptr_pos = 0;
    for (int32_t i = 0; i < 3; i++)
    {
        float* _tem  =  new float[this->grids[i*2] * this->grids[i*2 + 1] * 3 * colum];
        // memcpy(_tem, data_head, this->grids[i*2] * this->grids[i*2 + 1] * (this->nc + 5) * sizeof(float));
        // data_head = data_head + this->grids[i*2] * this->grids[i*2 + 1] * (this->nc + 5);
        for (int j = 0; j < this->grids[i*2] * this->grids[i*2 + 1] * 3; j++)
        {
            // *_temp = (((int32_t)pOutputTensor->data.uint8[i * colum + j]) - zero_point) * scale;
            for (int k = 0; k < colum; k++) {
                _tem[j + k] = (float)pOutputTensor->data.f[ptr_pos + j * colum + k];
            }
        }
        ptr_pos += this->grids[i*2] * this->grids[i*2 + 1] * 3 * colum;
        std::cout << "ptr position " << ptr_pos << std::endl;
        v.push_back(_tem);
    }
    std::cout << "end of tensorToFloatPtr2D" << std::endl;
    return v;
}

void YOLOV5::nonMaximumSupprition(
    std::vector<std::vector<float>> &predV,
    const int &row,
    const int &colum,
    std::vector<cv::Rect> &boxes,
    std::vector<float> &confidences,
    std::vector<int> &classIds,
    std::vector<int> &indices)

{

    std::vector<cv::Rect> boxesNMS;
    int max_wh = 40960;
    std::vector<float> scores;
    double confidence;
    cv::Point classId;

    for (int i = 0; i < row; i++)
    {
        if (predV[i][4] > conf_thres)
        {
            // height--> image.rows,  width--> image.cols;
            int left = (predV[i][0] - predV[i][2] / 2) * _img_width;
            int top = (predV[i][1] - predV[i][3] / 2) * _img_height;
            int w = predV[i][2] * _img_width;
            int h = predV[i][3] * _img_height;

            for (int j = 5; j < colum; j++)
            {
                // # conf = obj_conf * cls_conf
                scores.push_back(predV[i][j] * predV[i][4]);
            }

            cv::minMaxLoc(scores, 0, &confidence, 0, &classId);
            scores.clear();
            int c = classId.x * max_wh;
            if (confidence > conf_thres)
            {
                boxes.push_back(cv::Rect(left, top, w, h));
                confidences.push_back(confidence);
                classIds.push_back(classId.x);
                boxesNMS.push_back(cv::Rect(left, top, w, h));
            }
        }
    }
    // float eta = 0;
    // int top_k = 0;
    cv::dnn::NMSBoxes(boxesNMS, confidences, conf_thres, iou_thre, indices);
}

void YOLOV5::run(cv::Mat frame, Prediction &out_pred, std::vector<Anchor> &filtered_outputs)
{
    g_print("yolov5 inference running !!\n");
    _img_height = frame.rows;
    _img_width = frame.cols;

    g_print("fact image height %d, width %d\n",_img_height, _img_width);
    cv::Mat model_input;
    cv::Mat img_(_img_height, _img_width, CV_8UC3, frame.data);
    // preprocess(frame);
    PreProcess(model_input, img_);
    g_print("preprocess finished \n");
    g_print("fact input model`s image height %d, width %d\n",model_input.rows, model_input.cols);
    cv::imwrite("/data/fact_img.png", model_input);
    _input_8 = model_input.isContinuous()? model_input.data: model_input.clone().data;
    // fill(_input_8, model_input);

    g_print("yolov5 inference preprocess and fill data finished !!\n");
    // Inference
    auto start = std::chrono::high_resolution_clock::now();
    TfLiteStatus status = _interpreter->Invoke();
    if (status != kTfLiteOk)
    {
        std::cout << "\nFailed to run inference!!\n";
        exit(1);
    }
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    g_print("Model run time 'milliseconds' except post process : %ld\n",duration.count());

    g_print("yolov5 inference finished !!\n");

    int _out = _interpreter->outputs()[0];
    TfLiteIntArray *_out_dims = _interpreter->tensor(_out)->dims;
    int _out_row   = _out_dims->data[1];   // 25200
    int _out_colum = _out_dims->data[2];   // class number + 5 ---> 85     bbox cond class

    std::cout << "output row = " << _out_row << ", output colcum = " << _out_colum << std::endl;
    // int _out_type  = _interpreter->tensor(_out)->type;

    g_print("get output data after yolov5 inference!!\n");
    std::cout << "get output size " << _interpreter->outputs().size() << std::endl;
    TfLiteTensor *pOutputTensor = _interpreter->tensor(_interpreter->outputs()[0]);
#if 0
    std::cout << "convert output tensor to vector 2D" << std::endl;
    std::vector<std::vector<float>> predV = tensorToVector2D(pOutputTensor, _out_row, _out_colum);
 
    std::vector<int> indices;
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    g_print("nms for output data!!\n");
    nonMaximumSupprition(predV, _out_row, _out_colum, boxes, confidences, classIds, indices);

    g_print("fill out_pred boxes info!!\n");
    for (int i = 0; i < indices.size(); i++)
    {
        out_pred.boxes.push_back(boxes[indices[i]]);
        out_pred.scores.push_back(confidences[indices[i]]);
        out_pred.labels.push_back(classIds[indices[i]]);
    }
#endif
    std::cout << "convert output tensor to float ptr" << std::endl;
    std::cout << "output row = " << _out_row << ", output colcum = " << _out_colum << std::endl;
    std::vector<float*> network_outputs = tensorToFloatPtr2D(pOutputTensor, _out_row, _out_colum);
    std::cout << "post process with using anchor" << std::endl;
    PostProcess(filtered_outputs, network_outputs);

    std::cout << "first filter size: " << filtered_outputs.size() << std::endl;
}

static inline float sigmoid(float input){
  return 1.0 / (1.0 + exp(-input));
}

static void nms_cpu(std::vector<Anchor>& boxes, float threshold, std::vector<Anchor>& filterOutBoxes) {

	if (boxes.size() == 0)
		return;
	std::vector<size_t> idx(boxes.size());

	for (unsigned i = 0; i < idx.size(); i++)
	{
		idx[i] = i;
	}

	sort(boxes.begin(), boxes.end(), std::greater<Anchor>());

	while (idx.size() > 0)
	{
		int good_idx = idx[0];
		filterOutBoxes.push_back(boxes[good_idx]);

		std::vector<size_t> tmp = idx;
		idx.clear();
		for (unsigned i = 1; i < tmp.size(); i++)
		{
			int tmp_i = tmp[i];
			float inter_x1 = std::max(boxes[good_idx].finalbox_.xmin_, boxes[tmp_i].finalbox_.xmin_);
			float inter_y1 = std::max(boxes[good_idx].finalbox_.ymin_, boxes[tmp_i].finalbox_.ymin_);
			float inter_x2 = std::min(boxes[good_idx].finalbox_.xmax_, boxes[tmp_i].finalbox_.xmax_);
			float inter_y2 = std::min(boxes[good_idx].finalbox_.ymax_, boxes[tmp_i].finalbox_.ymax_);

			float w = std::max((inter_x2 - inter_x1 + 1), 0.0F);
			float h = std::max((inter_y2 - inter_y1 + 1), 0.0F);

			float inter_area = w * h;
			float area_1 = (boxes[good_idx].finalbox_.xmax_ - boxes[good_idx].finalbox_.xmin_ + 1) * (boxes[good_idx].finalbox_.ymax_ - boxes[good_idx].finalbox_.ymin_ + 1);
			float area_2 = (boxes[tmp_i].finalbox_.xmax_ - boxes[tmp_i].finalbox_.xmin_ + 1) * (boxes[tmp_i].finalbox_.ymax_ - boxes[tmp_i].finalbox_.ymin_ + 1);
			float index = inter_area / (area_1 + area_2 - inter_area);
			if (index <= threshold)
				idx.push_back(tmp_i);
		}
	}
}

int YOLOV5::PostProcess(std::vector<Anchor>& filtered_outputs,std::vector<float*> network_outputs)
{
  filtered_outputs.clear();
  std::map<int, std::vector<Anchor>> map_anchors;
  // std::vector<Anchor> tmp_anchors;
  int grid_w, grid_h;
  for(int index = 0; index < 3; ++index){
    grid_w = this->grids[index*2];
    grid_h = this->grids[index*2+1];
    for(int a=0; a<3; ++a){
      for(int y=0; y<grid_h;++y){
        for(int x=0; x < grid_w; ++x){

          int pos = a * grid_w * grid_h * (5 + nc) + y * grid_w * (5 + nc) + x * (5 + nc);
          float pred_x = sigmoid(network_outputs[index][0 + pos]);
          float pred_y = sigmoid(network_outputs[index][1 + pos]);
          float pred_w = sigmoid(network_outputs[index][2 + pos]);
          float pred_h = sigmoid(network_outputs[index][3 + pos]);
          float pred_conf = sigmoid(network_outputs[index][4 + pos]);
          if(pred_conf < this->conf_thres) 
            continue;

          float max_pred_conf = 0.;
          int max_pred_conf_idx = 0;

          for(int i=0;i < nc;i++){
            if(pred_conf < this->conf_thres) 
                continue;
            float pred_cla = sigmoid(network_outputs[index][i + 5 + pos]);
            float tmp_score = pred_conf * pred_cla;
            if(max_pred_conf < tmp_score){
              max_pred_conf_idx = i;
              max_pred_conf = tmp_score;
            }
          }

          if((max_pred_conf * pred_conf) < this->conf_thres){ //this->conf_thre_list[max_pred_conf_idx]){
            continue;
          }else{
            Anchor anchor;
            anchor.score_ = max_pred_conf;
            anchor.class_index = max_pred_conf_idx;
            float c_x = 1.0 * (pred_x * 2 - 0.5 + x) * this->model_shape[0] / grid_w;
            float c_y = 1.0 * (pred_y * 2 - 0.5 + y) * this->model_shape[0] / grid_w;
            float c_w = 4.0 * pred_w * pred_w * this->anchors[index * 3 * 2 + a * 2];
            float c_h = 4.0 * pred_h * pred_h * this->anchors[index * 3 * 2 + a * 2 +1];
            anchor.finalbox_.xmin_ = (c_x - 0.5 * c_w - pad_left) / scale;
            anchor.finalbox_.ymin_ = (c_y - 0.5 * c_h - pad_top) / scale;
            anchor.finalbox_.xmax_ = (c_x + 0.5 * c_w - pad_left) / scale;
            anchor.finalbox_.ymax_ = (c_y + 0.5 * c_h - pad_top) / scale;
            if(map_anchors.find(max_pred_conf_idx) == map_anchors.end()){
              std::vector<Anchor> tmp_anchor{anchor};
              map_anchors[max_pred_conf_idx] = tmp_anchor;
            } else {
              map_anchors[max_pred_conf_idx].push_back(anchor);
            }
          }
        }
      }
    }
  }
  for(int i=0;i<this->nc;i++){
    if(map_anchors.find(i) != map_anchors.end()){
      //std::cout << map_anchors[i].size() << std::endl;
      nms_cpu(map_anchors[i], this->iou_thre, filtered_outputs);
    }
  }
  return 0;
}

