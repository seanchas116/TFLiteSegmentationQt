#pragma once
#include <QMainWindow>

namespace tflite {
class Interpreter;
}

class MainWindow : public QMainWindow {
    Q_OBJECT
  public:
    MainWindow();
    ~MainWindow();

  private:
    std::unique_ptr<tflite::Interpreter> _interpreter;
};
