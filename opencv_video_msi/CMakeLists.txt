# This is a minimal CMakeLists file which includes the Yoctopuce C++ lib.
cmake_minimum_required (VERSION 3.10)
project (Demo)

SET(CMAKE_CXX_STANDARD 11)

find_package( OpenCV REQUIRED )
find_package( Threads REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

# Adds the executable called "demo" that is built from the source files "main.cpp".
add_executable (demo gstreamer.cpp)

target_link_libraries (demo LINK_PUBLIC ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT})