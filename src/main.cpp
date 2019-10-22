#include <QApplication>
#include <QWidget>

int main(int argc, char* argv[]) {
    QApplication::setAttribute(Qt::AA_EnableHighDpiScaling);

    QApplication app(argc, argv);
    QWidget window;
    window.resize(1024, 768);
    window.show();
    return app.exec();
}
