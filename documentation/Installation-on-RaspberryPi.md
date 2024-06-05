# Installing on Raspberry Pi

## Installing TensorFlow

Installing TensorFlow on the Raspberry Pi is a lot more complicated than it used to be as there is no longer an official package available. However there is still an unofficial distribution, which at least means we don't have to resort to building and installing from source.

	$ sudo apt install -y libhdf5-dev unzip pkg-config python3-pip cmake make git python-is-python3 wget patchelf
	$ python -m venv --system-site-packages ~/.python-tf
	$ source ~/.python-tf/bin/activate
	$ pip install numpy==1.26.2 
	$ pip install keras_applications==1.0.8 --no-deps 
	$ pip install keras_preprocessing==1.1.2 --no-deps 
	$ pip install h5py==3.10.0 
	$ pip install pybind11==2.9.2 
	$ pip install packaging 
	$ pip install protobuf==3.20.3 
	$ pip install six wheel mock gdown 
	$ TFVER=2.15.0.post1
	$ PYVER=311
	$ ARCH=`python -c 'import platform; print(platform.machine())'`
	$ pip install --no-cache-dir https://github.com/PINTO0309/Tensorflow-bin/releases/download/v${TFVER}/tensorflow-${TFVER}-cp${PYVER}-none-linux_${ARCH}.whl

## Installing TensorFlow Lite

There is still an official TensorFlow Lite runtime package available for Raspberry Pi, so installation is much more simple than for full TensorFlow where that option is no longer available.

	$ python -m venv --system-site-packages ~/.python-tflite
	$ source ~/.python-tflite/bin/activate
	$ pip install opencv-python
	$ pip install tflite-runtime

