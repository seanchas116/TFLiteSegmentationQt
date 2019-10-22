#include "MainWindow.hpp"
#include <QApplication>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/model.h>

MainWindow::MainWindow() {
    auto modelPath = QApplication::applicationDirPath() + "/resources/deeplabv3_257_mv_gpu.tflite";
    auto model = tflite::FlatBufferModel::BuildFromFile(modelPath.toUtf8().data());
}
