#include "MainWindow.hpp"
#include <QApplication>
#include <QFileDialog>
#include <QLabel>
#include <QToolBar>
#include <QtDebug>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>

namespace {

// From https://github.com/tensorflow/examples/blob/master/lite/examples/image_segmentation/ios/ImageSegmentation/ImageSegmentator.swift
std::vector<QRgb> legendColorList = {
    0xFFB300, // Vivid Yellow
    0x803E75, // Strong Purple
    0xFF6800, // Vivid Orange
    0xA6BDD7, // Very Light Blue
    0xC10020, // Vivid Red
    0xCEA262, // Grayish Yellow
    0x817066, // Medium Gray
    0x007D34, // Vivid Green
    0xF6768E, // Strong Purplish Pink
    0x00538A, // Strong Blue
    0xFF7A5C, // Strong Yellowish Pink
    0x53377A, // Strong Violet
    0xFF8E00, // Vivid Orange Yellow
    0xB32851, // Strong Purplish Red
    0xF4C800, // Vivid Greenish Yellow
    0x7F180D, // Strong Reddish Brown
    0x93AA00, // Vivid Yellowish Green
    0x593315, // Deep Yellowish Brown
    0xF13A13, // Vivid Reddish Orange
    0x232C16, // Dark Olive Green
    0x00A1C2, // Vivid Blue
};

} // namespace

MainWindow::MainWindow() {
    // Build UI
    {
        _imageLabel = new QLabel();
        setCentralWidget(_imageLabel);

        auto toolBar = new QToolBar();
        toolBar->addAction(tr("Load Image..."), this, [this] {
            auto path = QFileDialog::getOpenFileName(this, "Open Image", "", "Image Files (*.png *.jpg *.bmp)");
            if (path.isEmpty()) {
                return;
            }

            QFile file(path);
            file.open(QIODevice::ReadOnly);
            auto image = QImage::fromData(file.readAll());
            loadImage(image);
        });
        addToolBar(toolBar);
    }

    // Load model
    {
        auto modelPath = QApplication::applicationDirPath() + "/resources/deeplabv3_257_mv_gpu.tflite";
        auto model = tflite::FlatBufferModel::BuildFromFile(modelPath.toUtf8().data());

        tflite::ops::builtin::BuiltinOpResolver resolver;
        tflite::InterpreterBuilder(*model.get(), resolver)(&_interpreter);

        _interpreter->AllocateTensors();

        auto inputTensor = _interpreter->tensor(_interpreter->inputs()[0]);
        int inputBatchSize = inputTensor->dims->data[0];
        int inputWidth = inputTensor->dims->data[1];
        int inputHeight = inputTensor->dims->data[2];
        int inputChannelCount = inputTensor->dims->data[3];

        qDebug() << "input type:" << inputTensor->type;
        qDebug() << "input batch size:" << inputBatchSize;
        qDebug() << "input width:" << inputWidth;
        qDebug() << "input height:" << inputHeight;
        qDebug() << "input channel count:" << inputChannelCount;

        auto outputTensor = _interpreter->tensor(_interpreter->outputs()[0]);
        int outputBatchSize = outputTensor->dims->data[0];
        int outputWidth = outputTensor->dims->data[1];
        int outputHeight = outputTensor->dims->data[2];
        int outputChannelCount = outputTensor->dims->data[3];

        qDebug() << "output type:" << outputTensor->type;
        qDebug() << "output batch size:" << outputBatchSize;
        qDebug() << "output width:" << outputWidth;
        qDebug() << "output height:" << outputHeight;
        qDebug() << "output channel count:" << outputChannelCount;
    }
}

MainWindow::~MainWindow() {
}

void MainWindow::loadImage(const QImage &image) {
    auto inputTensor = _interpreter->tensor(_interpreter->inputs()[0]);
    int inputWidth = inputTensor->dims->data[1];
    int inputHeight = inputTensor->dims->data[2];

    auto tensorData = _interpreter->typed_input_tensor<float>(_interpreter->inputs()[0]);

    auto inputImage = image.convertToFormat(QImage::Format_RGB888).scaled(inputWidth, inputHeight);
    float *dst = tensorData;
    for (int y = 0; y < inputHeight; ++y) {
        uint8_t *src = inputImage.scanLine(y);
        for (int x = 0; x < inputWidth * 3; ++x) {
            *dst++ = *src++ / 255.f;
        }
    }

    _interpreter->Invoke();

    _imageLabel->setPixmap(QPixmap::fromImage(image));
}
