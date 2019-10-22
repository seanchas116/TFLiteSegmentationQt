cmake_minimum_required(VERSION 3.15)
project(tflite-qt-app)

# Qt
find_package(Qt5Widgets REQUIRED)

set(CMAKE_AUTOMOC ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror")

include_directories("vendor/tensorflow-lite/include")

if(APPLE)
    link_directories("vendor/tensorflow-lite/osx_x86_64/lib")
endif()

FILE(GLOB_RECURSE SOURCES "src/*.cpp")
FILE(GLOB_RECURSE HEADERS "src/*.hpp")

add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

target_link_libraries(${PROJECT_NAME} Qt5::Widgets -ltensorflow-lite)

file(INSTALL ${CMAKE_CURRENT_SOURCE_DIR}/resources DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
