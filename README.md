# TFLiteSegmentationQt
Run TensorFlow Lite image segmentation in Qt desktop app

## Install tensorflow-lite

This repo already includes tensorflow-lite build (v2.0.0), but if you want to use tensorflow-lite built by yourself do the following:

* Clone [tensorflow repo](https://github.com/tensorflow/tensorflow)
* Build tensorflow-lite for desktop

```
cd path/to/tensorflow
./tensorflow/lite/tools/make/download_dependencies.sh
./tensorflow/lite/tools/make/build_lib.sh
```

* Copy .a and .h

```
# Copy TensorFlow Lite headers
cd path/to/tensorflow/tensorflow/lite
gcp --parents **/*.h /path/to/tflite-qt-app/vendor/tensorflow-lite/include/tensorflow/lite

# Copy flatbuffers headers
cp -r tools/make/downloads/flatbuffers/include/flatbuffers /path/to/tflite-qt-app/vendor/tensorflow-lite/include

# Copy libtensorflow-lite.a (platform name may vary)
cp tools/make/gen/osx_x86_64/lib/libtensorflow-lite.a /path/to/tflite-qt-app/vendor/tensorflow-lite/osx_x86_64/lib
```

