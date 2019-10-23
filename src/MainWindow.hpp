#pragma once
#include <QMainWindow>

namespace tflite {
class FlatBufferModel;
class Interpreter;
} // namespace tflite

class QLabel;

class MainWindow : public QMainWindow {
    Q_OBJECT
  public:
    MainWindow();
    ~MainWindow();

  private:
    void loadImage(const QImage &image);

    std::unique_ptr<tflite::FlatBufferModel> _model;
    std::unique_ptr<tflite::Interpreter> _interpreter;
    QLabel *_imageLabel;
};
