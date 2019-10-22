#include <QApplication>
#include <QWidget>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/model.h>

int main(int argc, char *argv[]) {
    QApplication::setAttribute(Qt::AA_EnableHighDpiScaling);
    QApplication app(argc, argv);

    auto modelPath = QApplication::applicationDirPath() + "/resources/deeplabv3_257_mv_gpu.tflite";
    auto model = tflite::FlatBufferModel::BuildFromFile(modelPath.toUtf8().data());

    QWidget window;
    window.resize(1024, 768);
    window.show();
    return app.exec();
}
