#pragma once
#include <QMainWindow>

namespace tflite {
class Interpreter;
}

class QLabel;

class MainWindow : public QMainWindow {
    Q_OBJECT
  public:
    MainWindow();
    ~MainWindow();

  private:
    void loadImage(const QImage &image);

    std::unique_ptr<tflite::Interpreter> _interpreter;
    QLabel *_imageLabel;
};
