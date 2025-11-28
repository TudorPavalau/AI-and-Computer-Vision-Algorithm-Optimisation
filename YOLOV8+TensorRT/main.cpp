#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>

// OpenCV
#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>

// TensorRT & CUDA Headers
#include <NvInfer.h>
#include <cuda_runtime_api.h>

using namespace nvinfer1;

// ==================================================================
// 1. UTILITARE
// ==================================================================
class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kERROR) std::cout << "[TRT] " << msg << std::endl;
    }
} gLogger;

struct Detection {
    int class_id;
    float confidence;
    cv::Rect box;
};

// Functie pentru calcul IoU (Intersection over Union)
float compute_iou(const cv::Rect& boxA, const cv::Rect& boxB) {
    cv::Rect inter = boxA & boxB;
    float areaInter = (float)inter.area();
    float areaUnion = (float)(boxA.area() + boxB.area() - areaInter);
    return (areaUnion > 0) ? (areaInter / areaUnion) : 0.0f;
}

// ==================================================================
// 2. CLASA TENSORRT OPTIMIZATA
// ==================================================================
class YoloTRT {
public:
    YoloTRT(const std::string& enginePath) {
        loadEngine(enginePath);
    }

    ~YoloTRT() {
        for (void* ptr : buffers) cudaFree(ptr);
        delete context; delete engine; delete runtime;
    }

    std::vector<Detection> detect(const cv::Mat& img, float conf_thres, float iou_thres, int num_classes) {
        // 1. Pre-procesare (Letterbox Resize)
        // YOLO vrea imaginea centrata in 640x640, pastrand proportiile
        cv::Mat blob = preprocess_img(img, inputDims.width, inputDims.height);

        // 2. Copiere CPU -> GPU
        cudaMemcpyAsync(buffers[inputIndex], blob.ptr<float>(), inputSize, cudaMemcpyHostToDevice, stream);

        // 3. Setare tensori (TRT 10)
        for (int i = 0; i < engine->getNbIOTensors(); ++i) {
            context->setTensorAddress(engine->getIOTensorName(i), buffers[i]);
        }

        // 4. Executie
        if (!context->enqueueV3(stream)) return {};

        // 5. Copiere GPU -> CPU
        std::vector<float> cpu_output(outputSize / sizeof(float));
        cudaMemcpyAsync(cpu_output.data(), buffers[outputIndex], outputSize, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        // 6. Post-procesare (Corectata cu unscaling)
        return postprocess(cpu_output, img.size(), conf_thres, iou_thres, num_classes);
    }

private:
    IRuntime* runtime = nullptr; ICudaEngine* engine = nullptr; IExecutionContext* context = nullptr;
    cudaStream_t stream; std::vector<void*> buffers;
    int inputIndex = -1, outputIndex = -1; size_t inputSize = 0, outputSize = 0;
    cv::Size inputDims = { 640, 640 };

    void loadEngine(const std::string& path) {
        std::ifstream file(path, std::ios::binary);
        if (!file.good()) throw std::runtime_error("Engine not found");
        file.seekg(0, file.end); size_t size = file.tellg(); file.seekg(0, file.beg);
        std::vector<char> trtModelStream(size); file.read(trtModelStream.data(), size); file.close();
        runtime = createInferRuntime(gLogger);
        engine = runtime->deserializeCudaEngine(trtModelStream.data(), size);
        context = engine->createExecutionContext();
        cudaStreamCreate(&stream);

        int nbIOTensors = engine->getNbIOTensors();
        buffers.resize(nbIOTensors);
        for (int i = 0; i < nbIOTensors; ++i) {
            const char* name = engine->getIOTensorName(i);
            Dims dims = engine->getTensorShape(name);
            size_t vol = 1;
            for (int j = 0; j < dims.nbDims; j++) {
                int d = (int)dims.d[j];
                if (d == -1) d = 1;
                if (engine->getTensorIOMode(name) == TensorIOMode::kINPUT && d == -1) d = 640;
                vol *= d;
            }
            // Fix pentru output dinamic YOLOv8
            if (std::string(name).find("output") != std::string::npos && vol < 8400) vol = 1 * 100 * 8400;

            size_t sizeBytes = vol * sizeof(float);
            cudaMalloc(&buffers[i], sizeBytes);

            if (engine->getTensorIOMode(name) == TensorIOMode::kINPUT) {
                inputIndex = i; inputSize = sizeBytes;
                if (dims.nbDims >= 4) { inputDims.height = (int)dims.d[2]; inputDims.width = (int)dims.d[3]; }
            }
            else { outputIndex = i; outputSize = sizeBytes; }
        }
    }

    // Preprocesare corecta (Letterbox)
    cv::Mat preprocess_img(const cv::Mat& img, int target_w, int target_h) {
        float scale = std::min((float)target_w / img.cols, (float)target_h / img.rows);
        int new_w = (int)(img.cols * scale);
        int new_h = (int)(img.rows * scale);

        cv::Mat resized;
        cv::resize(img, resized, cv::Size(new_w, new_h));

        // Creare imagine neagra (canvas)
        cv::Mat canvas(target_h, target_w, CV_8UC3, cv::Scalar(114, 114, 114));
        // Centrare imagine
        int top = (target_h - new_h) / 2;
        int left = (target_w - new_w) / 2;
        resized.copyTo(canvas(cv::Rect(left, top, new_w, new_h)));

        // Pregatire pentru DNN (Normalize + SwapRB + CHW)
        cv::Mat blob;
        cv::dnn::blobFromImage(canvas, blob, 1.0 / 255.0, cv::Size(), cv::Scalar(), true, false);
        return blob;
    }

    // Postprocesare cu "Un-Letterbox" (Inversarea transformarii)
    std::vector<Detection> postprocess(const std::vector<float>& output, cv::Size originalSize, float conf_thres, float iou_thres, int nc) {
        std::vector<cv::Rect> boxes;
        std::vector<float> scores;
        std::vector<int> class_ids;

        int num_anchors = 8400;

        // Calculam factorii de scalare inversi (exact ca la preprocesare)
        float scale = std::min((float)inputDims.width / originalSize.width, (float)inputDims.height / originalSize.height);
        int new_w = (int)(originalSize.width * scale);
        int new_h = (int)(originalSize.height * scale);
        int pad_x = (inputDims.width - new_w) / 2;
        int pad_y = (inputDims.height - new_h) / 2;

        for (int i = 0; i < num_anchors; i++) {
            float max_score = -1.0f;
            int max_class_id = -1;
            for (int c = 0; c < nc; c++) {
                float score = output[(4 + c) * num_anchors + i];
                if (score > max_score) { max_score = score; max_class_id = c; }
            }

            if (max_score >= conf_thres) {
                float cx = output[0 * num_anchors + i];
                float cy = output[1 * num_anchors + i];
                float w = output[2 * num_anchors + i];
                float h = output[3 * num_anchors + i];

                // --- MATEMATICA CRITICA: UN-LETTERBOX ---
                // Scadem padding-ul si impartim la scara
                float x_original = (cx - pad_x) / scale;
                float y_original = (cy - pad_y) / scale;
                float w_original = w / scale;
                float h_original = h / scale;

                // Din centru in colt stanga-sus
                int left = (int)(x_original - w_original / 2);
                int top = (int)(y_original - h_original / 2);
                int width = (int)w_original;
                int height = (int)h_original;

                // Clamp la dimensiunile imaginii (sa nu iasa din poza)
                left = std::max(0, std::min(left, originalSize.width - 1));
                top = std::max(0, std::min(top, originalSize.height - 1));
                width = std::min(width, originalSize.width - left);
                height = std::min(height, originalSize.height - top);

                boxes.push_back(cv::Rect(left, top, width, height));
                scores.push_back(max_score);
                class_ids.push_back(max_class_id);
            }
        }

        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, scores, conf_thres, iou_thres, indices);

        std::vector<Detection> results;
        for (int idx : indices) {
            results.push_back({ class_ids[idx], scores[idx], boxes[idx] });
        }
        return results;
    }
};

// ==================================================================
// 3. MAIN (TUNED FOR ACCURACY)
// ==================================================================
int main() {
    std::string det_path = "yolov8n.engine";
    std::string ang_path = "angle_detector.engine";
    std::string img_path = "test.png";

    // 1. LISTA OBIECTE (COCO Standard - 80 clase)
    std::vector<std::string> obj_classes = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"
    };    
    // Unghiuri
    std::vector<std::string> ang_labels = {
        "0",    // Clasa 0
        "135",  // Clasa 1 (Aici gresisem inainte)
        "180",  // Clasa 2
        "225",  // Clasa 3
        "270",  // Clasa 4
        "315",  // Clasa 5
        "45",   // Clasa 6
        "90"    // Clasa 7
    };

    // 3. MAPARE UNGHIURI (VALORI MATEMATICE) - Aceeasi ordine pentru calculul sagetii
    std::vector<float> ang_values = {
        0.0f,   // Clasa 0
        135.0f, // Clasa 1
        180.0f, // Clasa 2
        225.0f, // Clasa 3
        270.0f, // Clasa 4
        315.0f, // Clasa 5
        45.0f,  // Clasa 6
        90.0f   // Clasa 7
    };

    try {
        std::cout << "Initializare TensorRT..." << std::endl;
        YoloTRT detector(det_path);
        YoloTRT angle_detector(ang_path);

        cv::Mat img = cv::imread(img_path);
        if (img.empty()) { std::cerr << "Imaginea nu exista!" << std::endl; return -1; }

        std::cout << "Rulare Inferenta..." << std::endl;

        // --- SCHIMBARE: PRAGURI MAI MICI PENTRU A GASI TOT ---
        // 0.25 Confidenta este standardul YOLOv8 pentru vizualizare
        auto det_results = detector.detect(img, 0.25f, 0.45f, 8);
        auto ang_results = angle_detector.detect(img, 0.25f, 0.45f, 8);

        std::cout << "Obiecte gasite: " << det_results.size() << std::endl;

        for (const auto& obj : det_results) {
            std::string label = (obj.class_id < obj_classes.size()) ? obj_classes[obj.class_id] : "Obj";

            // Matching IoU
            int best_ang_idx = -1;
            float max_iou = 0.0f;

            for (const auto& ang : ang_results) {
                float iou = compute_iou(obj.box, ang.box);
                // Daca se suprapun macar 30%
                if (iou > 0.3f && iou > max_iou) {
                    max_iou = iou;
                    best_ang_idx = ang.class_id;
                }
            }

            if (best_ang_idx != -1 && best_ang_idx < ang_labels.size()) {
                label += " (" + ang_labels[best_ang_idx] + ")";
            }

            // Desenare
            cv::rectangle(img, obj.box, cv::Scalar(0, 255, 0), 2);
            cv::putText(img, label, cv::Point(obj.box.x, obj.box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);

            if (best_ang_idx != -1 && best_ang_idx < ang_values.size()) {
                cv::Point center = (obj.box.tl() + obj.box.br()) * 0.5;
                double rad = ang_values[best_ang_idx] * (CV_PI / 180.0);
                cv::Point endP;
                endP.x = (int)(center.x + 40.0 * cos(rad));
                endP.y = (int)(center.y - 40.0 * sin(rad)); // Y inversat
                cv::arrowedLine(img, center, endP, cv::Scalar(255, 0, 0), 2, 8, 0, 0.3);
            }
        }

        cv::imshow("Rezultat Final Corectat", img);
        cv::waitKey(0);

    }
    catch (const std::exception& e) {
        std::cerr << "CRASH: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}