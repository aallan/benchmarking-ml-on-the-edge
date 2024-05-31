#!/usr/bin/env python3

import platform
PLATFORM = platform.system().lower()
GOOGLE = 'edge_tpu'
INTEL = 'ncs2'
NVIDIA = 'jetson_nano'
PI = 'raspberry_pi'
MAC = 'darwin'
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
                import tensorflow.contrib.tensorrt
                print("TensorFlow with GPU support present.")
                print("Assuming Jetson Nano.")
                PLATFORM = NVIDIA
            else:
                print("No GPU support in TensorFlow.")
        except ImportError:
            print("No TensorFlow support found.")

LEGAL_PLATFORMS = NVIDIA, PI, MAC
assert PLATFORM in LEGAL_PLATFORMS, "Don't understand platform %s." % PLATFORM

import sys
import os
import logging as log
import argparse
import subprocess
from timeit import default_timer as timer

import cv2

from PIL import Image
from PIL import ImageFont, ImageDraw

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

def inference_tf(runs, image, model, output, label=None):
   if label:
       labels = ReadLabelFile(label)
   else:
       labels = None
   
   tf_config = tf.ConfigProto()
   tf_config.gpu_options.allow_growth = True
   
   with tf.gfile.FastGFile(model, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        
        
   with tf.Session(config=tf_config) as sess:
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')
        
        img = Image.open(image)
        draw = ImageDraw.Draw(img, 'RGBA')
        helvetica=ImageFont.truetype("./Helvetica.ttf", size=72)
        
        picture = cv2.imread(image)
        initial_h, initial_w, channels = picture.shape
        frame = cv2.resize(picture, (300, 300))
        frame = frame[:, :, [2, 1, 0]]  # BGR2RGB
        frame = frame.reshape(1, frame.shape[0], frame.shape[1], 3)
        
        #  Start synchronous inference and get inference result
        # Run inference.
        print("Running inferencing for ", runs, " times.")
       
        if runs == 1:
           start = timer()
           out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                           sess.graph.get_tensor_by_name('detection_scores:0'),
                           sess.graph.get_tensor_by_name('detection_boxes:0'),
                           sess.graph.get_tensor_by_name('detection_classes:0')],
                          feed_dict={'image_tensor:0': frame})
           end = timer()
           print('Elapsed time is ', ((end - start)/runs)*1000, 'ms' )
        else:
           start = timer()
           print('Initial run, discarding.')
           out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                           sess.graph.get_tensor_by_name('detection_scores:0'),
                           sess.graph.get_tensor_by_name('detection_boxes:0'),
                           sess.graph.get_tensor_by_name('detection_classes:0')],
                          feed_dict={'image_tensor:0': frame})
           end = timer()
           print('First run time is ', (end - start)*1000, 'ms')
           start = timer()
           for i in range(runs):
              out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                              sess.graph.get_tensor_by_name('detection_scores:0'),
                              sess.graph.get_tensor_by_name('detection_boxes:0'),
                              sess.graph.get_tensor_by_name('detection_classes:0')],
                             feed_dict={'image_tensor:0': frame})
           end = timer()
           print('Elapsed time is ', ((end - start)/runs)*1000, 'ms' )
        
        # Visualize detected bounding boxes.
        num_detections = int(out[0][0])
        for i in range(num_detections):
            classId = int(out[3][0][i])
            score = float(out[1][0][i])
            bbox = [float(v) for v in out[2][0][i]]
            if score > 0.5:
                xmin = bbox[1] * initial_w
                ymin = bbox[0] * initial_h
                xmax = bbox[3] * initial_w
                ymax = bbox[2] * initial_h
                if labels:
                   print(labels[classId], 'score = ', score)
                else:
                   print ('score = ', score)
                box = [xmin, ymin, xmax, ymax]
                print( 'box = ', box )
                draw_rectangle(draw, box, (0,128,128,20), width=5)
                if labels:
                   draw.text((box[0] + 20, box[1] + 20), labels[classId], fill=(255,255,255,20), font=helvetica)
        img.save(output)
        print ('Saved to ', output)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='Path of the detection model.', required=True)
    parser.add_argument('--label', help='Path of the labels file.')
    parser.add_argument('--input', help='File path of the input image.', required=True)
    parser.add_argument('--output', help='File path of the output image.')
    parser.add_argument('--runs', help='Number of times to run the inference', type=int, default=1)
    args = parser.parse_args()
    
    if ( args.output):
      output_file = args.output
    else:
      output_file = 'out.jpg'
    
    if ( args.label ):
      label_file = args.label
    else:
      label_file = None
    
    result = inference_tf( args.runs, args.input, args.model, output_file, label_file)

if __name__ == '__main__':
  main()

