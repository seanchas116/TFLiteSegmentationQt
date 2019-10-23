#include "MainWindow.hpp"
#include <QApplication>
#include <QFileDialog>
#include <QLabel>
#include <QPainter>
#include <QToolBar>
#include <QtDebug>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>

namespace {

// From https://github.com/tensorflow/examples/blob/master/lite/examples/image_segmentation/ios/ImageSegmentation/ImageSegmentator.swift
std::vector<QRgb> legendColorList = {
    0xFFFFB300, // Vivid Yellow
    0xFF803E75, // Strong Purple
    0xFFFF6800, // Vivid Orange
    0xFFA6BDD7, // Very Light Blue
    0xFFC10020, // Vivid Red
    0xFFCEA262, // Grayish Yellow
    0xFF817066, // Medium Gray
    0xFF007D34, // Vivid Green
    0xFFF6768E, // Strong Purplish Pink
    0xFF00538A, // Strong Blue
    0xFFFF7A5C, // Strong Yellowish Pink
    0xFF53377A, // Strong Violet
    0xFFFF8E00, // Vivid Orange Yellow
    0xFFB32851, // Strong Purplish Red
    0xFFF4C800, // Vivid Greenish Yellow
    0xFF7F180D, // Strong Reddish Brown
    0xFF93AA00, // Vivid Yellowish Green
    0xFF593315, // Deep Yellowish Brown
    0xFFF13A13, // Vivid Reddish Orange
    0xFF232C16, // Dark Olive Green
    0xFF00A1C2, // Vivid Blue
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
        _model = tflite::FlatBufferModel::BuildFromFile(modelPath.toUtf8().data());

        tflite::ops::builtin::BuiltinOpResolver resolver;
        tflite::InterpreterBuilder(*_model, resolver)(&_interpreter);

        if (_interpreter->AllocateTensors() != kTfLiteOk) {
            qDebug() << "Error AllocateTensors";
        }

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

    auto tensorData = _interpreter->typed_tensor<float>(_interpreter->inputs()[0]);

    // TODO: transpose?
    auto inputImage = image.convertToFormat(QImage::Format_RGB888).scaled(inputWidth, inputHeight);
    float *dst = tensorData;
    for (int y = 0; y < inputHeight; ++y) {
        uint8_t *src = inputImage.scanLine(y);
        for (int x = 0; x < inputWidth * 3; ++x) {
            *dst++ = *src++ / 255.f;
        }
    }

    if (_interpreter->Invoke() != kTfLiteOk) {
        qDebug() << "Error Invoke";
    }

    auto outputData = _interpreter->typed_tensor<float>(_interpreter->outputs()[0]);
    QImage outputImage(inputWidth, inputHeight, QImage::Format_RGB888);
    for (int y = 0; y < inputHeight; ++y) {
        for (int x = 0; x < inputWidth; ++x) {
            const int classCount = 21;
            int index = std::max_element(outputData, outputData + classCount) - outputData;
            qDebug() << index;
            outputData += classCount;
            outputImage.setPixel(x, y, legendColorList.at(index));
        }
    }

    QImage displayImage = inputImage;
    QPainter painter(&displayImage);
    painter.setOpacity(0.5);
    painter.drawImage(displayImage.rect(), outputImage, outputImage.rect());

    //_imageLabel->setPixmap(QPixmap::fromImage(image));
    _imageLabel->setPixmap(QPixmap::fromImage(displayImage));
    //_imageLabel->setPixmap(QPixmap::fromImage(inputImage));
}
