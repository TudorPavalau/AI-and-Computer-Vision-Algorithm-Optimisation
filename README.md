# 🚀 YOLOv8 Inference Optimization with Libtorch & TensorRT

> High-performance C++ inference pipelines for YOLOv8 object detection, optimized using **LibTorch** and **NVIDIA TensorRT** for real-time computer vision applications.

---

## 📋 Overview

This repository provides two production-ready C++ implementations for running YOLOv8 object detection inference with maximum performance:

- **YOLOv8 + LibTorch** — Direct PyTorch C++ API integration for flexible, cross-platform deployment
- **YOLOv8 + TensorRT** — NVIDIA GPU-accelerated inference using TensorRT engine optimization, achieving significant speedups over standard PyTorch inference

Both implementations are designed for developers who need to integrate state-of-the-art object detection into C++ applications with minimal overhead and maximum throughput.

---

## 📁 Project Structure

```
AI-and-Computer-Vision-Algorithm-Optimisation/
│
├── YOLOV8+Libtorch/          # YOLOv8 inference via PyTorch C++ API (LibTorch)
│   ├── CMakeLists.txt
│   └── main.cpp
│
├── YOLOV8+TensorRT/          # YOLOv8 inference via NVIDIA TensorRT
│   ├── CMakeLists.txt
│   └── main.cpp
│
└── README.md
```

---

## ⚡ Performance

TensorRT optimization can yield **2x–5x speedup** over standard PyTorch inference on modern NVIDIA GPUs, depending on the model size and precision (FP32, FP16, INT8).

| Backend      | Framework      | Typical Latency (YOLOv8n) | GPU Required |
|--------------|----------------|---------------------------|--------------|
| LibTorch     | PyTorch C++ API | ~15–20 ms/frame           | Optional     |
| TensorRT     | NVIDIA TensorRT | ~5–8 ms/frame             | ✅ Yes        |

---

## 🛠️ Prerequisites

### Common Requirements

- CMake >= 3.18
- C++17 compatible compiler (GCC 9+, MSVC 2019+, Clang 10+)
- YOLOv8 model weights (`.pt` or exported format)

### For YOLOv8 + LibTorch

- [LibTorch](https://pytorch.org/get-started/locally/) (CPU or CUDA build)
- OpenCV >= 4.5

### For YOLOv8 + TensorRT

- NVIDIA GPU with CUDA support
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) >= 11.x
- [cuDNN](https://developer.nvidia.com/cudnn)
- [TensorRT](https://developer.nvidia.com/tensorrt) >= 8.x
- OpenCV >= 4.5

---

## 🔧 Build Instructions

### Clone the Repository

```bash
git clone https://github.com/TudorPavalau/AI-and-Computer-Vision-Algorithm-Optimisation.git
cd AI-and-Computer-Vision-Algorithm-Optimisation
```

### Build YOLOv8 + LibTorch

```bash
cd "YOLOV8+Libtorch"
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH=/path/to/libtorch
cmake --build . --config Release
```

### Build YOLOv8 + TensorRT

```bash
cd "YOLOV8+TensorRT"
mkdir build && cd build
cmake .. \
  -DCMAKE_PREFIX_PATH=/path/to/tensorrt \
  -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
cmake --build . --config Release
```

---

## 🚀 Usage

### Export YOLOv8 Model

Before running inference, export your YOLOv8 model to the appropriate format using Ultralytics:

```python
from ultralytics import YOLO

# For LibTorch — export to TorchScript
model = YOLO("yolov8n.pt")
model.export(format="torchscript")

# For TensorRT — export to ONNX first
model.export(format="onnx")
```

Then convert ONNX to TensorRT engine:

```bash
trtexec --onnx=yolov8n.onnx --saveEngine=yolov8n.engine --fp16
```

### Run Inference

```bash
# LibTorch
./yolov8_libtorch --model yolov8n.torchscript --source image.jpg

# TensorRT
./yolov8_tensorrt --model yolov8n.engine --source image.jpg
```

---

## 🧠 How It Works

### LibTorch Pipeline

1. Load a TorchScript model (`.pt` / `.torchscript`)
2. Preprocess input image (resize, normalize, convert to tensor)
3. Run forward pass via `torch::jit::script::Module`
4. Post-process output (decode bounding boxes, apply NMS)

### TensorRT Pipeline

1. Load a serialized TensorRT engine (`.engine`)
2. Allocate GPU memory buffers using CUDA
3. Preprocess and transfer image to GPU
4. Execute inference via TensorRT execution context
5. Transfer results back to CPU and apply post-processing

---

## 📦 Dependencies Summary

| Dependency   | Version    | Purpose                      |
|--------------|------------|------------------------------|
| LibTorch     | >= 2.0     | PyTorch C++ inference API    |
| TensorRT     | >= 8.x     | GPU-optimized inference      |
| OpenCV       | >= 4.5     | Image loading & preprocessing|
| CUDA         | >= 11.x    | GPU computing                |
| CMake        | >= 3.18    | Build system                 |

---

## 📌 Notes

- The TensorRT engine file is hardware-specific — it must be rebuilt for each target GPU.
- FP16 precision is recommended for a balance between speed and accuracy.
- For edge deployment (e.g., NVIDIA Jetson), ensure TensorRT is installed via JetPack SDK.


---

## 👥 Authors

**Tudor Pavalau** — [GitHub Profile](https://github.com/TudorPavalau)

**Georgi Emanuel** — [GitHub Profile](https://github.com/GeorgiEmanuel)
