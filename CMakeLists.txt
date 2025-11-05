cmake_minimum_required(VERSION 3.18)

project(torchtest)

set(CMAKE_PREFIX_PATH "D:/libtorch/libtorch")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Torch REQUIRED)

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")

if (MSVC)
file(GLOB TORCH_DLLS "${CMAKE_PREFIX_PATH}/lib/*.dll")
add_custom_command(TARGET ${PROJECT_NAME}
POST_BUILD
COMMAND ${CMAKE_COMMAND} -E copy_if_different
${TORCH_DLLS}
$<TARGET_FILE_DIR:${PROJECT_NAME}>)
endif()
