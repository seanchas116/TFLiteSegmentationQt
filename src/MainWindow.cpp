#include "MainWindow.hpp"
#include <QApplication>
#include <QFileDialog>
#include <QLabel>
#include <QToolBar>
#include <QtDebug>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>

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

        qDebug() << outputTensor->dims->size;
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
