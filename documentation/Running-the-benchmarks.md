# Running the benchmarks

## Benchmarking TensorFlow

### Linux and macOS

The `benchmark_tf.py` script is used to run TensorFlow benchmarks on Linux (including Raspberry Pi) and macOS. This script can also used — with a TensorFlow installation which includes GPU support — on Nvidia Jetson hardware.

	$ source ~/.python-tf/bin/activate
	$ ./benchmark_tf.py --model PATH_TO_MODEL_FILE --label PATH_TO_LABEL_FILE --input INPUT_IMAGE --output LABELLED_OUTPUT_IMAGE --runs 10000

For example on a Raspberry Pi, benchmarking with the MobileNet v2 model for 10,000 inference runs the invocation would be,

	$ ./benchmark_tf.py --model ssd_mobilenet_v2/tf_for_linux_and_macos/frozen_inference_graph.pb --label ssd_mobilenet_v2/tf_for_linux_and_macos/coco_labels.txt --input fruit.jpg --output output.jpg --runs 10000

this will output an `output.jpg` image with the two objects (the banana and the apple) labelled.

**NOTE:** The first run of the model will take considerably longer than subsequent runs, with inferencing of ×10 of greater than normal inferencing. If the command line flag `--runs` is passed a number greater than one, the first run of the model will be discarded from subsequent timings.

### Nvidia GPU

While "native" TensorFlow models can be used on Nvidia Jetson hardware, inferencing is much quicker if the models have been optimised using Nvidia's own TensorRT framework. The `benchmark_tf_trt.py` script is used to run the benchmark on Nvidia GPU hardware using these optimised models.

## Benchmarking TensorFlow Lite

### Linux and macOS

The `benchmark_tf_lite.py` script is used to run TensorFlow Lite benchmarks on Linux (inluding Raspberry Pi) and macOS. 

	$ source ~/.python-tf-lite/bin/activate
	$ ./benchmark_tf_lite.py --model PATH_TO_MODEL_FILE --label PATH_TO_LABEL_FILE --input INPUT_IMAGE --output LABELLED_OUTPUT_IMAGE --runs 10000	

**NOTE:** Models passed to this script must be quantized. As for the TensorFlow script the first inferencing run is discarded from the calculation of the average inferencing time if `--runs` is greater than one.

## Benchmarking on the Edge TPU

TBD

## Benchmarking on Movidius and Intel NCS

TBD
