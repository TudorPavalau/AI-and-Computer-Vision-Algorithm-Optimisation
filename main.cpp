int main() {
    // --- A. SETARI FISIERE ---
    std::string det_obj_path = "yolov8n.torchscript";       // Detector Obiecte
    std::string det_ang_path = "angle_detector.torchscript"; // Detector Unghiuri
    std::string image_path = "test.png";

    float conf_threshold = 0.50;
    float iou_threshold = 0.45;

    // 1. LISTA OBIECTE
    std::vector<std::string> obj_classes = {
        "car", "van", "truck", "pedestrian",
        "person_sitting", "cyclist", "tram", "misc"
    };

    // 2. LISTA UNGHIURI (ETICHETE TEXT) - Ordinea ta specifica
    std::vector<std::string> angle_labels = {
        "Est (0)",      // Clasa 0
        "N-Vest (135)", // Clasa 1
        "Vest (180)",   // Clasa 2
        "S-Vest (225)", // Clasa 3
        "Sud (270)",    // Clasa 4
        "S-Est (315)",  // Clasa 5
        "N-Est (45)",   // Clasa 6
        "Nord (90)"     // Clasa 7
    };

    // 3. MAPARE UNGHIURI (VALORI MATEMATICE) - Corespunde index cu index cu lista de mai sus
    std::vector<float> class_to_degree = {
        0.0f,   // Clasa 0 -> 0 grade
        135.0f, // Clasa 1 -> 135 grade
        180.0f, // Clasa 2 -> 180 grade
        225.0f, // Clasa 3 -> 225 grade
        270.0f, // Clasa 4 -> 270 grade
        315.0f, // Clasa 5 -> 315 grade
        45.0f,  // Clasa 6 -> 45 grade
        90.0f   // Clasa 7 -> 90 grade
    };

    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    std::cout << "Device: " << (torch::cuda::is_available() ? "CUDA (GPU)" : "CPU") << std::endl;

    try {
        // --- Incarcare Modele ---
        torch::jit::script::Module model_obj = torch::jit::load(det_obj_path, device);
        model_obj.eval(); model_obj.to(device, torch::kFloat32);

        torch::jit::script::Module model_ang = torch::jit::load(det_ang_path, device);
        model_ang.eval(); model_ang.to(device, torch::kFloat32);

        // --- Incarcare Imagine ---
        cv::Mat image = cv::imread(image_path);
        if (image.empty()) return -1;

        // --- Pre-procesare ---
        cv::Mat input_img;
        letterbox(image, input_img, { 640, 640 });
        cv::cvtColor(input_img, input_img, cv::COLOR_BGR2RGB);
        input_img.convertTo(input_img, CV_32FC3, 1.0 / 255.0);
        torch::Tensor tensor = torch::from_blob(input_img.data, { input_img.rows, input_img.cols, 3 });
        tensor = tensor.permute({ 2, 0, 1 }).unsqueeze(0).to(device);

        // --- Inferenta ---
        torch::NoGradGuard no_grad;

        // Detector Obiecte
        torch::Tensor out_obj = model_obj.forward({ tensor }).toTensor().cpu();
        auto keep_obj = non_max_suppression(out_obj, 0.4, 0.45)[0];

        // Detector Unghiuri
        torch::Tensor out_ang = model_ang.forward({ tensor }).toTensor().cpu();
        auto keep_ang = non_max_suppression(out_ang, 0.4, 0.45)[0];

        // --- Procesare ---
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

                // Matching IoU
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

                    // Luam textul din vectorul tau personalizat
                    std::string ang_str = (ang_class_id >= 0 && ang_class_id < angle_labels.size()) ? angle_labels[ang_class_id] : "?";
                    label_text += " [" + ang_str + "]";
                }

                // 1. Desenam Cutia
                cv::Point p1(b_obj[0], b_obj[1]);
                cv::Point p2(b_obj[2], b_obj[3]);
                cv::rectangle(image, p1, p2, cv::Scalar(0, 255, 0), 2);
                cv::putText(image, label_text, cv::Point(p1.x, p1.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);

                // 2. Desenam Sageata (Folosind MAPAREA MATEMATICA)
                if (ang_class_id >= 0 && ang_class_id < class_to_degree.size()) {
                    cv::Point center((p1.x + p2.x) / 2, (p1.y + p2.y) / 2);

                    // AICI FOLOSIM VECTORUL TAU DE GRADE
                    double angle_deg = class_to_degree[ang_class_id];
                    double angle_rad = angle_deg * (CV_PI / 180.0);

                    int len = 50;
                    cv::Point endP;

                    // X = cos, Y = -sin (pt ca Y e inversat in imagine)
                    endP.x = center.x + (int)(len * cos(angle_rad));
                    endP.y = center.y - (int)(len * sin(angle_rad));

                    cv::arrowedLine(image, center, endP, cv::Scalar(0, 0, 255), 3, 8, 0, 0.3);
                }
            }

            cv::imshow("Final Result Custom", image);
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