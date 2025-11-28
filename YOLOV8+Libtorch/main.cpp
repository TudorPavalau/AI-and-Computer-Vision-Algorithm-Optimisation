#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <torch/torch.h>
#include <torch/script.h>

using torch::indexing::Slice;
using torch::indexing::None;


float compute_iou(const float* boxA, const float* boxB) {
    float xA = std::max(boxA[0], boxB[0]);
    float yA = std::max(boxA[1], boxB[1]);
    float xB = std::min(boxA[2], boxB[2]);
    float yB = std::min(boxA[3], boxB[3]);

    float interArea = std::max(0.0f, xB - xA) * std::max(0.0f, yB - yA);
    if (interArea == 0) return 0.0f;

    float boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]);
    float boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]);

    return interArea / (boxAArea + boxBArea - interArea);
}

float generate_scale(cv::Mat& image, const std::vector<int>& target_size) {
    int origin_w = image.cols; int origin_h = image.rows;
    int target_h = target_size[0]; int target_w = target_size[1];
    float ratio_h = static_cast<float>(target_h) / static_cast<float>(origin_h);
    float ratio_w = static_cast<float>(target_w) / static_cast<float>(origin_w);
    return std::min(ratio_h, ratio_w);
}

float letterbox(cv::Mat& input_image, cv::Mat& output_image, const std::vector<int>& target_size) {
    if (input_image.cols == target_size[1] && input_image.rows == target_size[0]) {
        if (input_image.data == output_image.data) { return 1.; }
        else { output_image = input_image.clone(); return 1.; }
    }
    float resize_scale = generate_scale(input_image, target_size);
    int new_shape_w = (int)std::round(input_image.cols * resize_scale);
    int new_shape_h = (int)std::round(input_image.rows * resize_scale);
    float padw = (target_size[1] - new_shape_w) / 2.0f; float padh = (target_size[0] - new_shape_h) / 2.0f;
    int top = (int)std::round(padh - 0.1); int bottom = (int)std::round(padh + 0.1);
    int left = (int)std::round(padw - 0.1); int right = (int)std::round(padw + 0.1);
    cv::resize(input_image, output_image, cv::Size(new_shape_w, new_shape_h), 0, 0, cv::INTER_AREA);
    cv::copyMakeBorder(output_image, output_image, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(114., 114., 114));
    return resize_scale;
}

torch::Tensor xyxy2xywh(const torch::Tensor& x) {
    auto y = torch::empty_like(x);
    y.index_put_({ "...", 0 }, (x.index({ "...", 0 }) + x.index({ "...", 2 })).div(2));
    y.index_put_({ "...", 1 }, (x.index({ "...", 1 }) + x.index({ "...", 3 })).div(2));
    y.index_put_({ "...", 2 }, x.index({ "...", 2 }) - x.index({ "...", 0 }));
    y.index_put_({ "...", 3 }, x.index({ "...", 3 }) - x.index({ "...", 1 }));
    return y;
}

torch::Tensor xywh2xyxy(const torch::Tensor& x) {
    auto y = torch::empty_like(x);
    auto dw = x.index({ "...", 2 }).div(2); auto dh = x.index({ "...", 3 }).div(2);
    y.index_put_({ "...", 0 }, x.index({ "...", 0 }) - dw); y.index_put_({ "...", 1 }, x.index({ "...", 1 }) - dh);
    y.index_put_({ "...", 2 }, x.index({ "...", 0 }) + dw); y.index_put_({ "...", 3 }, x.index({ "...", 1 }) + dh);
    return y;
}

torch::Tensor nms(const torch::Tensor& bboxes, const torch::Tensor& scores, float iou_threshold) {
    if (bboxes.numel() == 0) return torch::empty({ 0 }, bboxes.options().dtype(torch::kLong));
    auto x1_t = bboxes.select(1, 0).contiguous(); auto y1_t = bboxes.select(1, 1).contiguous();
    auto x2_t = bboxes.select(1, 2).contiguous(); auto y2_t = bboxes.select(1, 3).contiguous();
    torch::Tensor areas_t = (x2_t - x1_t) * (y2_t - y1_t);
    auto order_t = std::get<1>(scores.sort(true, 0, true));
    auto ndets = bboxes.size(0);
    torch::Tensor suppressed_t = torch::zeros({ ndets }, bboxes.options().dtype(torch::kByte));
    torch::Tensor keep_t = torch::zeros({ ndets }, bboxes.options().dtype(torch::kLong));
    auto suppressed = suppressed_t.data_ptr<uint8_t>(); auto keep = keep_t.data_ptr<int64_t>();
    auto order = order_t.data_ptr<int64_t>();
    auto x1 = x1_t.data_ptr<float>(); auto y1 = y1_t.data_ptr<float>();
    auto x2 = x2_t.data_ptr<float>(); auto y2 = y2_t.data_ptr<float>();
    auto areas = areas_t.data_ptr<float>();
    int64_t num_to_keep = 0;
    for (int64_t _i = 0; _i < ndets; _i++) {
        auto i = order[_i]; if (suppressed[i] == 1) continue;
        keep[num_to_keep++] = i;
        auto ix1 = x1[i]; auto iy1 = y1[i]; auto ix2 = x2[i]; auto iy2 = y2[i]; auto iarea = areas[i];
        for (int64_t _j = _i + 1; _j < ndets; _j++) {
            auto j = order[_j]; if (suppressed[j] == 1) continue;
            auto xx1 = std::max(ix1, x1[j]); auto yy1 = std::max(iy1, y1[j]);
            auto xx2 = std::min(ix2, x2[j]); auto yy2 = std::min(iy2, y2[j]);
            auto w = std::max(static_cast<float>(0), xx2 - xx1); auto h = std::max(static_cast<float>(0), yy2 - yy1);
            auto inter = w * h; auto ovr = inter / (iarea + areas[j] - inter);
            if (ovr > iou_threshold) suppressed[j] = 1;
        }
    }
    return keep_t.narrow(0, 0, num_to_keep);
}

torch::Tensor non_max_suppression(torch::Tensor& prediction, float conf_thres = 0.25, float iou_thres = 0.45, int max_det = 300) {
    auto bs = prediction.size(0);
    auto nc = prediction.size(1) - 4; auto nm = prediction.size(1) - nc - 4; auto mi = 4 + nc;
    auto xc = prediction.index({ Slice(), Slice(4, mi) }).amax(1) > conf_thres;
    prediction = prediction.transpose(-1, -2);
    prediction.index_put_({ "...", Slice({None, 4}) }, xywh2xyxy(prediction.index({ "...", Slice(None, 4) })));
    std::vector<torch::Tensor> output;
    for (int i = 0; i < bs; i++) output.push_back(torch::zeros({ 0, 6 + nm }, prediction.device()));
    for (int xi = 0; xi < prediction.size(0); xi++) {
        auto x = prediction[xi]; x = x.index({ xc[xi] });
        auto x_split = x.split({ 4, nc, nm }, 1); auto box = x_split[0], cls = x_split[1], mask = x_split[2];
        auto [conf, j] = cls.max(1, true);
        x = torch::cat({ box, conf, j.toType(torch::kFloat), mask }, 1);
        x = x.index({ conf.view(-1) > conf_thres });
        if (!x.size(0)) continue;
        auto c = x.index({ Slice(), Slice{5, 6} }) * 7680;
        auto boxes = x.index({ Slice(), Slice(None, 4) }) + c; auto scores = x.index({ Slice(), 4 });
        auto i = nms(boxes, scores, iou_thres); i = i.index({ Slice(None, max_det) });
        output[xi] = x.index({ i });
    }
    return torch::stack(output);
}

torch::Tensor scale_boxes(const std::vector<int>& img1_shape, torch::Tensor& boxes, const std::vector<int>& img0_shape) {
    auto gain = (std::min)((float)img1_shape[0] / img0_shape[0], (float)img1_shape[1] / img0_shape[1]);
    auto pad0 = std::round((float)(img1_shape[1] - img0_shape[1] * gain) / 2. - 0.1);
    auto pad1 = std::round((float)(img1_shape[0] - img0_shape[0] * gain) / 2. - 0.1);
    boxes.index_put_({ "...", 0 }, boxes.index({ "...", 0 }) - pad0); boxes.index_put_({ "...", 2 }, boxes.index({ "...", 2 }) - pad0);
    boxes.index_put_({ "...", 1 }, boxes.index({ "...", 1 }) - pad1); boxes.index_put_({ "...", 3 }, boxes.index({ "...", 3 }) - pad1);
    boxes.index_put_({ "...", Slice(None, 4) }, boxes.index({ "...", Slice(None, 4) }).div(gain));
    return boxes;
}


int main() {
    std::string det_obj_path = "yolov8n.torchscript";       // Detector Obiecte
    std::string det_ang_path = "angle_detector.torchscript"; // Detector Unghiuri
    std::string image_path = "test.png";

    float conf_threshold = 0.50;
    float iou_threshold = 0.45;

    std::vector<std::string> obj_classes = {
        "car", "van", "truck", "pedestrian",
        "person_sitting", "cyclist", "tram", "misc"
    };

    std::vector<std::string> angle_labels = {
        "0",    // Clasa 0
        "135",  // Clasa 1
        "180",  // Clasa 2
        "225",  // Clasa 3
        "270",  // Clasa 4
        "315",  // Clasa 5
        "45",   // Clasa 6
        "90"    // Clasa 7
    };

    std::vector<float> class_to_degree = {
        0.0f,   // Clasa 0
        135.0f, // Clasa 1
        180.0f, // Clasa 2
        225.0f, // Clasa 3
        270.0f, // Clasa 4
        315.0f, // Clasa 5
        45.0f,  // Clasa 6
        90.0f   // Clasa 7
    };

    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    std::cout << "Device: " << (torch::cuda::is_available() ? "CUDA (GPU)" : "CPU") << std::endl;

    try {
        torch::jit::script::Module model_obj = torch::jit::load(det_obj_path, device);
        model_obj.eval(); model_obj.to(device, torch::kFloat32);

        torch::jit::script::Module model_ang = torch::jit::load(det_ang_path, device);
        model_ang.eval(); model_ang.to(device, torch::kFloat32);

        cv::Mat image = cv::imread(image_path);
        if (image.empty()) return -1;

        cv::Mat input_img;

        letterbox(image, input_img, { 640, 640 });
        cv::cvtColor(input_img, input_img, cv::COLOR_BGR2RGB);
        input_img.convertTo(input_img, CV_32FC3, 1.0 / 255.0);
        torch::Tensor tensor = torch::from_blob(input_img.data, { input_img.rows, input_img.cols, 3 });
        tensor = tensor.permute({ 2, 0, 1 }).unsqueeze(0).to(device);

        torch::NoGradGuard no_grad;

        // Detector Obiecte
        torch::Tensor out_obj = model_obj.forward({ tensor }).toTensor().cpu();
        auto keep_obj = non_max_suppression(out_obj, 0.4, 0.45)[0];

        // Detector Unghiuri
        torch::Tensor out_ang = model_ang.forward({ tensor }).toTensor().cpu();
        auto keep_ang = non_max_suppression(out_ang, 0.4, 0.45)[0];

        if (keep_obj.size(0) > 0) {
            auto boxes_obj = keep_obj.index({ Slice(), Slice(None, 4) });
            scale_boxes({ input_img.rows, input_img.cols }, boxes_obj, { image.rows, image.cols });

            if (keep_ang.size(0) > 0) {
                auto boxes_ang = keep_ang.index({ Slice(), Slice(None, 4) });
                scale_boxes({ input_img.rows, input_img.cols }, boxes_ang, { image.rows, image.cols });
            }

            for (int i = 0; i < keep_obj.size(0); i++) {
                float* b_obj = (float*)boxes_obj[i].data_ptr();
                int cls_obj = keep_obj[i][5].item().toInt();

                std::string label_text = (cls_obj >= 0 && cls_obj < obj_classes.size()) ? obj_classes[cls_obj] : "Obj";

                int best_ang_idx = -1;
                float best_iou = 0.0f;

                for (int j = 0; j < keep_ang.size(0); j++) {
                    float* b_ang = (float*)keep_ang.index({ j, Slice(None, 4) }).data_ptr();
                    float iou = compute_iou(b_obj, b_ang);
                    if (iou > 0.5 && iou > best_iou) {
                        best_iou = iou;
                        best_ang_idx = j;
                    }
                }

                int ang_class_id = -1;
                if (best_ang_idx != -1) {
                    ang_class_id = keep_ang[best_ang_idx][5].item().toInt();

              
                    std::string ang_str = (ang_class_id >= 0 && ang_class_id < angle_labels.size()) ? angle_labels[ang_class_id] : "?";
                    label_text += " (" + ang_str + ")";
                }

                cv::Point p1(b_obj[0], b_obj[1]);
                cv::Point p2(b_obj[2], b_obj[3]);
                cv::rectangle(image, p1, p2, cv::Scalar(0, 255, 0), 2);
                cv::putText(image, label_text, cv::Point(p1.x, p1.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);

                if (ang_class_id >= 0 && ang_class_id < class_to_degree.size()) {
                    cv::Point center((p1.x + p2.x) / 2, (p1.y + p2.y) / 2);

                    double angle_deg = class_to_degree[ang_class_id];
                    double angle_rad = angle_deg * (CV_PI / 180.0);

                    int len = 50;
                    cv::Point endP;

                    endP.x = center.x + (int)(len * cos(angle_rad));
                    endP.y = center.y - (int)(len * sin(angle_rad));

                    cv::arrowedLine(image, center, endP, cv::Scalar(0, 0, 255), 3, 8, 0, 0.3);
                }
            }

            cv::imshow("Final Result", image);
            cv::waitKey(0);
        }
        else {
            std::cout << "Nimic detectat." << std::endl;
        }

    }
    catch (const std::exception& e) {
        std::cerr << "Eroare: " << e.what() << std::endl;
    }
    return 0;
}