#!/usr/bin/env python3

import platform
PLATFORM = platform.system().lower()
GOOGLE = 'edge_tpu'
INTEL = 'ncs2'
NVIDIA = 'jetson_nano'
PI = 'raspberry_pi'
IS_LINUX = (PLATFORM == 'linux')

if IS_LINUX:
    PLATFORM = platform.linux_distribution()[0].lower()
    if PLATFORM == 'debian':
        try:
            with open('/proc/cpuinfo') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('Hardware') and ( line.endswith('BCM2708') or line.endswith('BCM2835')):
                        PLATFORM = PI
                        print("Running on a Raspberry Pi.")
                        break
        except:
            print("Unknown platform based on Debian.")
            pass
    elif PLATFORM == 'mendel':
        PLATFORM = GOOGLE
        print("Running on a Coral Dev Board.")

try:
    from edgetpu.detection.engine import DetectionEngine
    print("DetectionEngine present.")
    PLATFORM = GOOGLE
except ImportError:
    try:
        from openvino.inference_engine import IENetwork, IEPlugin
        print("OpenVINO present.")
        print("Assuming Movidius hardware.")
        PLATFORM = INTEL
    except ImportError:
        try:
            import tensorflow as tf
            if (tf.test.is_built_with_cuda()):
                print("TensorFlow with GPU support present.")
                print("Assuming Jetson Nano.")
                PLATFORM = NVIDIA
            else:
                print("No GPU support in TensorFlow.")
        except ImportError:
            print("No TensorFlow support found.")

LEGAL_PLATFORMS = INTEL
assert PLATFORM in LEGAL_PLATFORMS, "This version of the script is intended for Intel hardware."

import os
import sys
import argparse
from timeit import default_timer as timer

import cv2
from PIL import Image
from PIL import ImageFont, ImageDraw, ImageColor

# Function to draw a rectangle with width > 1
def draw_rectangle(draw, coordinates, color, width=1):
    for i in range(width):
        rect_start = (coordinates[0] - i, coordinates[1] - i)
        rect_end = (coordinates[2] + i, coordinates[3] + i)
        draw.rectangle((rect_start, rect_end), outline = color, fill = color)

# Function to read labels from text files.
def ReadLabelFile(file_path):
  with open(file_path, 'r') as f:
    lines = f.readlines()
  ret = {}
  for line in lines:
    pair = line.strip().split(maxsplit=1)
    ret[int(pair[0])] = pair[1].strip()
  return ret

def inference_openvino(runs, image, output, model, weights, label=None):
    # See https://software.intel.com/en-us/articles/transitioning-from-intel-movidius-neural-compute-sdk-to-openvino-toolkit
    
    if label:
       labels = ReadLabelFile(label)
    else:
       labels = None
    
    # Open image.
    img = Image.open(image)
    draw = ImageDraw.Draw(img, 'RGBA')
    helvetica=ImageFont.truetype("./Helvetica.ttf", size=72)
    
    #  Plugin initialization for specified device and load extensions library if specified
    plugin = IEPlugin(device="MYRIAD") 
    
    #  Read in Graph file (IR)
    net = IENetwork(model=model, weights=weights) 
    
    #assert len(net.inputs.keys()) == 1, "Demo supports only single input topologies"
    #assert len(net.outputs) == 1, "Demo supports only single output topologies"
    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))
    
    #  Load network to the plugin
    exec_net = plugin.load(network=net)
    
    #  Obtain and preprocess input tensor (image)
    #  Read and pre-process input image  maybe we don't need to show these details
    picture = cv2.imread(image)
    initial_h, initial_w, channels = picture.shape
    
    #  Preprocessing is neural network dependent maybe we don't show this
    n, c, h, w = net.inputs[input_blob].shape
    frame = cv2.resize(picture, (w, h))
    frame = frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
    frame = frame.reshape((n, c, h, w))
    
    #  Start synchronous inference and get inference result
    # Run inference.
    print("Running inferencing for ", runs, " times.")
    
    if runs == 1:
       start = timer()
       res = exec_net.infer(inputs={input_blob: frame})
       end = timer()
       print('Elapsed time is ', ((end - start)/runs)*1000, 'ms' )
    else:
       start = timer()
       print('Initial run, discarding.')
       res = exec_net.infer(inputs={input_blob: frame})
       end = timer()
       print('First run time is ', (end - start)*1000, 'ms')
       start = timer()
       for i in range(runs):
           res = exec_net.infer(inputs={input_blob: frame})
       end = timer()
       print('Elapsed time is ', ((end - start)/runs)*1000, 'ms' )
    
    
    if res:
       # Processing output blob
       # obj[1] = class, obj[2] = probability, obj3-6] = coordinates 
       #print(res)
       print("Processing output")
       res = res[out_blob]
       #print(res)
       for obj in res[0][0]:
           if ( obj[2] > 0.6):
              #print("obj =",obj)
              xmin = int(obj[3] * initial_w)
              ymin = int(obj[4] * initial_h)
              xmax = int(obj[5] * initial_w)
              ymax = int(obj[6] * initial_h)
              class_id = int(obj[1])
              if labels:
                 print(labels[class_id], 'score = ', obj[2])
              else:
                 print ('score = ', obj[2])
              box = [xmin, ymin, xmax, ymax]
              print( 'box = ', box )
              draw_rectangle(draw, box, (128,128,0,20), width=5)
              if labels:
                 draw.text((box[0] + 20, box[1] + 20), labels[class_id], fill=(255,255,255,20), font=helvetica)
       img.save(output)
       print ('Saved to ', output)
    else:
      print ('No object detected!')
    
    del net
    del exec_net
    del plugin

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='Path of the detection model (XML file).', required=True)
    parser.add_argument('--label', help='Path of the labels file.')
    parser.add_argument('--input', help='File path of the input image.', required=True)
    parser.add_argument('--output', help='File path of the output image.')
    parser.add_argument('--runs', help='Number of times to run the inference', type=int, default=1)
    args = parser.parse_args()
    
    model_xml = args.model
    model_bin = os.path.splitext(model_xml)[0] + ".bin"
    
    if ( args.output):
      output_file = args.output
    else:
      output_file = 'out.jpg'
    
    if ( args.label ):
      label_file = args.label
    else:
      label_file = None
    
    inference_openvino(args.runs, args.input, args.output, model_xml, model_bin, label_file)


if __name__ == '__main__':
  main()

