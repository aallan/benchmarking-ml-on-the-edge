#!/usr/bin/env python3

import sys
import os
import logging as log
import argparse
import subprocess
from timeit import default_timer as timer

from PIL import Image
from PIL import ImageFont, ImageDraw

try:
    import xnornet
except ImportError:
    sys.exit("The xnornet wheel is not installed.  "
             "Please install it with pip:\n\n"
             "    python3 -m pip install --user xnornet-<...>.whl\n\n"
             "(drop the --user if you are using a virtualenv)")

# Function to draw a rectangle with width > 1
def draw_rectangle(draw, coordinates, color, width=1):
    for i in range(width):
        rect_start = (coordinates[0] - i, coordinates[1] - i)
        rect_end = (coordinates[2] + i, coordinates[3] + i)
        draw.rectangle((rect_start, rect_end), outline = color, fill = color)

def inference_xnor(runs, image, output):
   
   model = xnornet.Model.load_built_in()
   if model.result_type != xnornet.EvaluationResultType.BOUNDING_BOXES:
        sys.exit(model.name + " is not a detection model! This sample "
                 "requires a detection model to be installed (e.g. "
                 "person-pet-vehicle).")
   
   img = Image.open(image)
   draw = ImageDraw.Draw(img, 'RGBA')
   helvetica=ImageFont.truetype("./Helvetica.ttf", size=72)
        
   initial_w, initial_h = img.size
   picture = img.resize((300, 300))
        
   #  Start synchronous inference and get inference result
   # Run inference.
   print("Running inferencing for ", runs, " times.")
   
   if runs == 1:
      start = timer()	  
      boxes = model.evaluate(xnornet.Input.rgb_image(picture.size, picture.tobytes()))
      end = timer()
      print('Elapsed time is ', ((end - start)/runs)*1000, 'ms' )
   else:
      start = timer()
      print('Initial run, discarding.')
      boxes = model.evaluate(xnornet.Input.rgb_image(picture.size, picture.tobytes()))
      end = timer()
      print('First run time is ', (end - start)*1000, 'ms')
      start = timer()
      for i in range(runs):
         boxes = model.evaluate(xnornet.Input.rgb_image(picture.size, picture.tobytes()))
      end = timer()
      print('Elapsed time is ', ((end - start)/runs)*1000, 'ms' )
   
   
   print(boxes)
   for box in boxes:
      label = box.class_label.label
      xmin = box.rectangle.x * initial_w
      ymin = box.rectangle.y * initial_h
      xmax = (box.rectangle.x + box.rectangle.width)*initial_w
      ymax = (box.rectangle.y + box.rectangle.height)*initial_h
      bounds = [xmin, ymin, xmax, ymax]
      draw_rectangle(draw, bounds, (0,128,128,20), width=5)
      draw.text((bounds[0] + 20, bounds[1] + 20), label, fill=(255,255,255,20), font=helvetica)
   img.save(output)
   print ('Saved to ', output)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='File path of the input image.', required=True)
    parser.add_argument('--output', help='File path of the output image.')
    parser.add_argument('--runs', help='Number of times to run the inference', type=int, default=1)
    args = parser.parse_args()
    
    if ( args.output):
      output_file = args.output
    else:
      output_file = 'out.jpg'
    
    result = inference_xnor( args.runs, args.input, output_file)

if __name__ == '__main__':
  main()

