# Installing on macOS

## Installing TensorFlow

There is an official TensorFlow package for macOS. You can install it as below,

	$ brew install python
	$ python -m venv --system-site-packages ~/.python-tf
	$ source ~/.python-tf/bin/activate
	$ pip install tensorflow
	$ pip install opencv-python
	$ pip install pillow


## Installing TensorFlow Lite

**NOTE:** There is currently no `tflite-runtime` wheel for Python 3.12. See [this Github issue](https://github.com/tensorflow/tensorflow/issues/62003) for more information.