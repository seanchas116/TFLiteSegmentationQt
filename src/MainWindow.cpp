#include "MainWindow.hpp"
#include <QApplication>
#include <QtDebug>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>

MainWindow::MainWindow() {
    auto modelPath = QApplication::applicationDirPath() + "/resources/deeplabv3_257_mv_gpu.tflite";
    auto model = tflite::FlatBufferModel::BuildFromFile(modelPath.toUtf8().data());

    std::unique_ptr<tflite::Interpreter> interpreter;

    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder(*model.get(), resolver)(&interpreter);

    interpreter->AllocateTensors();

    auto inputTensor = interpreter->tensor(interpreter->inputs()[0]);
    int batchSize = inputTensor->dims->data[0];
    int imageWidth = inputTensor->dims->data[1];
    int imageHeight = inputTensor->dims->data[2];
    int imageChannelCount = inputTensor->dims->data[3];

    qDebug() << "batch size:" << batchSize;
    qDebug() << "width:" << imageWidth;
    qDebug() << "height:" << imageHeight;
    qDebug() << "channel count:" << imageChannelCount;
}
