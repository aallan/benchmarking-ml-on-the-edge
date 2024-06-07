# Benchmarking Machine Learning on the Edge

![Graph of benchmarked inferencing time in milli-seconds for the for MobileNet v2 model and the MobileNet v1 SSD 0.75 depth model, trained using the Common Objects in Context (COCO) dataset with an input size of 300×300.](documentation/images/graph.png "Inferencing time in milli-seconds for the for MobileNet v2 model (left hand bars, blue) and the MobileNet v1 SSD 0.75 depth model (right hand bars, green), trained using the Common Objects in Context (COCO) dataset with an input size of 300×300.")
_Inferencing time in milli-seconds for the for MobileNet v2 model (left hand bars, blue) and the MobileNet v1 SSD 0.75 depth model (right hand bars, green), trained using the Common Objects in Context (COCO) dataset with an input size of 300×300._

| Board  | Framework | Connection | MobileNet v2 (ms) | MobileNet v1 (ms) |
| --- | --- | --- | --- | --- |
| Jetson Nano  | TensorFlow  | | 309.3 | 276.0 |
| Jetson Nano  | TensorRT  | | 72.3 | 61.6 |
| Coral Dev Board | Edge TPU | | 20.9 | 15.7 |
| Coral USB Accelerator | Edge TPU | USB2 | 58.1 | 49.3 |
| Coral USB Accelerator | Edge TPU | USB3 | 18.2 | 14.9 |
| Movidius NCS | OpenVINO | USB2 | 204.5 | 115.7 |
| Movidius NCS | OpenVINO | USB3 | 176.4 | 88.4 |
| Intel NCS2 | OpenVIINO | USB2 | 118.6 | 87.2 |
| Intel NCS2 | OpenVINO | USB3 | 80.4 | 52.8 |
| Raspberry Pi 3, Model B+ | TensorFlow | | 654.0 | 480.3 |
| Raspberry Pi 4 | TensorFlow | | 483.5 | 263.9 |
| Raspberry Pi 5 | TensorFlow | | 148.9 | 66.2 |
| Raspberry Pi 3, Model B+ | TensorFlow Lite | | 379.6 | 271.5 |
| Raspberry Pi 4 | TensorFlow Lite | | 112.6 | 82.7 |
| Raspberry Pi 5 | TensorFlow Lite | | 23.5 | 16.9 |

The latest results are presented in the article [benchmarking the Raspberry Pi 5](https://www.hackster.io/news/benchmarking-tensorflow-and-tensorflow-lite-on-raspberry-pi-5-b9156d58a6a2).

**NOTE:** See the [documentation directory](documentation) for instructions on how to install TensorFlow and TensorFlow Lite, and how to run the benchmarking scripts.

## Getting Started with Google's Edge TPU

* [Hands on with the Coral Dev Board](https://medium.com/@aallan/hands-on-with-the-coral-dev-board-adbcc317b6af)
* [How to use a Raspberry Pi to flash the new firmware onto the Coral Dev Board](https://medium.com/@aallan/how-to-use-a-raspberry-pi-to-flash-new-firmware-onto-the-coral-dev-board-503aacf635b9)
* [Hands on with the Coral USB Accelerator](https://medium.com/@aallan/hands-on-with-the-coral-usb-accelerator-a37fcb323553)

**NOTE:** These guides are likely to be out of date and are in need of updating.

## Getting Started with Intel's Movidius

* [Getting started with the Intel Neural Compute Stick 2 and the Raspberry Pi](https://blog.hackster.io/getting-started-with-the-intel-neural-compute-stick-2-and-the-raspberry-pi-6904ccfe963)

**NOTE:** These guides are likely to be out of date and are in need of updating.

## Getting Started with Nvidia's GPUs

* [Getting started with the Nvidia Jetson Nano Developer Kit](https://blog.hackster.io/getting-started-with-the-nvidia-jetson-nano-developer-kit-43aa7c298797)

**NOTE:** These guides are likely to be out of date and are in need of updating.

## Benchmarking Machine Learning

* [The Big Benchmark Roundup](https://aallan.medium.com/the-big-benchmarking-roundup-a561fbfe8719)
* [Benchmarking Edge Computing](https://aallan.medium.com/benchmarking-edge-computing-ce3f13942245)
* [Benchmarking TensorFlow and TensorFlow Lite on the Raspberry Pi](https://blog.hackster.io/benchmarking-tensorflow-and-tensorflow-lite-on-the-raspberry-pi-43f51b796796)
* [Benchmarking the Xnor AI2GO Platform on the Raspberry Pi](https://blog.hackster.io/benchmarking-the-xnor-ai2go-platform-on-the-raspberry-pi-628a82af8aea)
* [Benchmarking Machine Learning on the new Raspberry Pi 4](https://blog.hackster.io/benchmarking-machine-learning-on-the-new-raspberry-pi-4-model-b-88db9304ce4)
* [Benchmarking TensorFlow Lite on the new Raspberry Pi 4](https://blog.hackster.io/benchmarking-tensorflow-lite-on-the-new-raspberry-pi-4-model-b-3fd859d05b98)
* [Benchmarking the Intel Neural Compute Stick on the new Raspberry Pi 4](https://blog.hackster.io/benchmarking-the-intel-neural-compute-stick-on-the-new-raspberry-pi-4-model-b-e419393f2f97)
* [Benchmarking TensorFlow and TensorFlow Lite on Raspberry Pi 5](https://www.hackster.io/news/benchmarking-tensorflow-and-tensorflow-lite-on-raspberry-pi-5-b9156d58a6a2) 

## TO DO

The benchmark code need to be updated to run the latest versions of the inferenecing frameworks:

- [ ] benchmark_edgetpu.py - Script for Edge TPU (Coral) hardware
- [ ] benchmark_intel.py - Script for Intel (Movidius) hardware using OpenVINO
- [x] benchmark_tf.py - Script for TensorFlow on generic hardware (CPU and GPU)
- [x] benchmark_tf_lite.py - Script for TensorFlow Lite on generic hardware
- [ ] benchmark_tf_trt.py - Script for Nvidia Jetson hardware using TensorRT

**NOTE:** The `benchmark_edgetpu.py` script currently uses the deprecated [edgetpu library](https://coral.ai/docs/edgetpu/api-intro/), and needs to be updated to use the [pycoral library](https://coral.ai/docs/reference/py/). However even that library is [no longer properly supported](https://github.com/google-coral/pycoral/issues/137) by Google.

## Licence

The code in this repository is licensed under the MIT licence.

Copyright &copy; 2019-2024 Alasdair Allan &lt;alasdair@babilim.co.uk&gt;

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.