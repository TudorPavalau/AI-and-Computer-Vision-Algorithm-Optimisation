cmake_minimum_required(VERSION 3.10)
project(TestProject)

set(CMAKE_CXX_STANDARD 17)

# --- 1. SETARI TENSORRT ---
# Aici am pus calea ta exacta (cu slash-uri inversate pt CMake)
set(TENSORRT_ROOT "C:/TensorRT-10.10.0.31") 

# Includem header-ele
include_directories(${TENSORRT_ROOT}/include)

# Gasim librariile (.lib)
# Cautam explicit numele cu sufixul _10 specific versiunii tale
find_library(NVINFER_LIB 
    NAMES nvinfer_10 nvinfer 
    PATHS ${TENSORRT_ROOT}/lib 
    NO_DEFAULT_PATH
)

find_library(NVONNXPARSER_LIB 
    NAMES nvonnxparser_10 nvonnxparser 
    PATHS ${TENSORRT_ROOT}/lib 
    NO_DEFAULT_PATH
)

# Verificare de siguranta (sa nu treaca mai departe daca nu le gaseste)
if(NOT NVINFER_LIB OR NOT NVONNXPARSER_LIB)
    message(FATAL_ERROR "Nu am gasit librariile TensorRT in ${TENSORRT_ROOT}/lib. Verifica daca exista nvinfer_10.lib acolo.")
endif()
# --------------------------------

# 2. Gasim pachetele standard
find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Fix pentru eroarea nvToolsExt
if(NOT TARGET CUDA::nvToolsExt)
    add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
endif()

find_package(Torch REQUIRED)

add_executable(test_app main.cpp)

# 3. Legam bibliotecile
target_link_libraries(test_app 
    ${OpenCV_LIBS} 
    ${TORCH_LIBRARIES} 
    CUDA::cudart
    ${NVINFER_LIB}       
    ${NVONNXPARSER_LIB}  
)

# 4. Setari Stiva
set_target_properties(test_app PROPERTIES LINK_FLAGS "/STACK:10000000")

# 5. Copiere automata DLL-uri
if (MSVC)
  # LibTorch DLLs
  file(COPY ${TORCH_INSTALL_PREFIX}/lib/caffe2_nvrtc.dll DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/Release)
  file(COPY ${TORCH_INSTALL_PREFIX}/lib/c10.dll DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/Release)
  file(COPY ${TORCH_INSTALL_PREFIX}/lib/torch_cpu.dll DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/Release)
  file(COPY ${TORCH_INSTALL_PREFIX}/lib/torch_cuda.dll DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/Release)
  file(COPY ${TORCH_INSTALL_PREFIX}/lib/torch.dll DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/Release)

  # TensorRT DLLs (Copiem tot ce e in folderul lib al TRT)
  file(GLOB TRT_DLLS "${TENSORRT_ROOT}/lib/*.dll")
  foreach(DLL ${TRT_DLLS})
      file(COPY ${DLL} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/Release)
  endforeach()
endif()