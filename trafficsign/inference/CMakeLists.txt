#Cmake lowest  version 
CMAKE_MINIMUM_REQUIRED(VERSION 3.4)

#project name
PROJECT(tflite_test CXX)

#allow gdb debug, add C++11 support and other options
set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS} -g -ftest-coverage -fprofile-arcs -Wno-deprecated")
set(CMAKE_CXX_FLAGS_DEBUG "-std=c++11 $ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "-std=c++11 $ENV{CXXFLAGS} -O3 -Wall")

add_definitions(-w)

# Try to find OpenCV
# set(OpenCV_DIR ....)
find_package(OpenCV REQUIRED)
if (OpenCV_FOUND)
    message(STATUS "    OpenCV library status:")
    message(STATUS "    version: ${OpenCV_VERSION}")
    message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}" \n)
else ()
    message(FATAL_ERROR "Could not locate OpenCV" \n)
endif()

#Tensorflow
set(TFLITE_LIBS tflite)
IF (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    MESSAGE(STATUS "current hardware platform: x86_64 ")
    set(TENSORFLOW_LIBRARY_DIRS "/home/thundersoft/software/tensorflow/bazel-bin/tensorflow/lite")
ELSEIF (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    MESSAGE(STATUS "current hardware platform: aarch64")
    set(TENSORFLOW_LIBRARY_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/libs/")
ENDIF ()

find_package(PkgConfig REQUIRED)
pkg_check_modules(GLIB           REQUIRED glib-2.0)
pkg_check_modules(GFLAGS         REQUIRED gflags)

# include file dir
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    $ENV{TENSORFLOW_ROOT}/
    ${GLIB_INCLUDE_DIRS}
    $ENV{TENSORFLOW_THIRD_PARTY}/
    ${GFLAGS_INCLUDE_DIRS}
    ../jsoncpp/include
)
 
file(GLOB SOURCE_FILES ./*.cpp)
add_executable(${PROJECT_NAME} 
    ${SOURCE_FILES}
)

MESSAGE(STATUS "yolov tflite test demo project name " ${PROJECT_NAME})
target_link_libraries( ${PROJECT_NAME}
    ${OpenCV_LIBS}
    ${TENSORFLOW_LIBRARY_DIRS}/libtensorflow-lite.so
    glib-2.0
    pthread
)