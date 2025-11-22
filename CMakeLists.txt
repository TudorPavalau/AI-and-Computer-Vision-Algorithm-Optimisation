cmake_minimum_required(VERSION 3.10)
project(TestProject)

set(CMAKE_CXX_STANDARD 17)

# 1. Gasim pachetele
find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Fix pentru eroarea nvToolsExt (Bypass)
if(NOT TARGET CUDA::nvToolsExt)
    add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
endif()

find_package(Torch REQUIRED)

add_executable(test_app main.cpp)

# Legam bibliotecile
target_link_libraries(test_app ${OpenCV_LIBS} ${TORCH_LIBRARIES} CUDA::cudart)

# 2. FIX CRITIC: Marim stiva (Stack Size) la 10MB ca sa nu dea Crash
set_target_properties(test_app PROPERTIES LINK_FLAGS "/STACK:10000000")

# 3. Copiere automata DLL-uri
if (MSVC)
  file(COPY ${TORCH_INSTALL_PREFIX}/lib/caffe2_nvrtc.dll DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/Release)
  file(COPY ${TORCH_INSTALL_PREFIX}/lib/c10.dll DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/Release)
  file(COPY ${TORCH_INSTALL_PREFIX}/lib/torch_cpu.dll DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/Release)
  file(COPY ${TORCH_INSTALL_PREFIX}/lib/torch_cuda.dll DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/Release)
  file(COPY ${TORCH_INSTALL_PREFIX}/lib/torch.dll DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/Release)
endif()