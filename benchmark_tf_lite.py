#!/usr/bin/env python3

import tensorflow as tf

import sys
import os
import logging as log
import argparse
import subprocess
from timeit import default_timer as timer

import cv2
import numpy as np

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
   
   # Load TFLite model and allocate tensors.
   interpreter = tf.lite.Interpreter(model_path=model)
   interpreter.allocate_tensors()
   
   # Get input and output tensors.
   input_details = interpreter.get_input_details()
   output_details = interpreter.get_output_details()
   height = input_details[0]['shape'][1]
   width = input_details[0]['shape'][2]
   floating_model = False
   if input_details[0]['dtype'] == np.float32:
       floating_model = True
   
   img = Image.open(image)
   draw = ImageDraw.Draw(img, 'RGBA')
   helvetica=ImageFont.truetype("./Helvetica.ttf", size=72)
        
   picture = cv2.imread(image)
   initial_h, initial_w, channels = picture.shape
   frame = cv2.resize(picture, (width, height))
   
   # add N dim
   input_data = np.expand_dims(frame, axis=0)
   
   if floating_model:
      input_data = (np.float32(input_data) - 127.5) / 127.5
   
   interpreter.set_num_threads(4)
   interpreter.set_tensor(input_details[0]['index'], input_data)
   
   #  Start synchronous inference and get inference result
   # Run inference.
   print("Running inferencing for ", runs, " times.")
       
   if runs == 1:
      start = timer()
      interpreter.invoke()
      end = timer()
      print('Elapsed time is ', ((end - start)/runs)*1000, 'ms' )
   else:
      start = timer()
      print('Initial run, discarding.')
      interpreter.invoke()
      end = timer()
      print('First run time is ', (end - start)*1000, 'ms')
      start = timer()
      for i in range(runs):
         interpreter.invoke()
      end = timer()
      print('Elapsed time is ', ((end - start)/runs)*1000, 'ms' )
        
   detected_boxes = interpreter.get_tensor(output_details[0]['index'])
   detected_classes = interpreter.get_tensor(output_details[1]['index'])
   detected_scores = interpreter.get_tensor(output_details[2]['index'])
   num_boxes = interpreter.get_tensor(output_details[3]['index'])
   
   #print("num_boxes:", num_boxes[0])
   #print("detected boxes:", detected_boxes)
   #print("detected classes:", detected_classes)
   #print("detected scores:", detected_scores)
   
   for i in range(int(num_boxes)):
      top, left, bottom, right = detected_boxes[0][i]
      classId = int(detected_classes[0][i])
      score = detected_scores[0][i]
      if score > 0.5:
          xmin = left * initial_w
          ymin = bottom * initial_h
          xmax = right * initial_w
          ymax = top * initial_h
          if labels:
              print(labels[classId], 'score = ', score)
          else:
              print ('score = ', score)
          box = [xmin, ymin, xmax, ymax]
          #print( 'box = ', box )
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

