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

LEGAL_PLATFORMS = GOOGLE
assert PLATFORM in LEGAL_PLATFORMS, "This version of the script is intended for Edge TPU hardware."

import argparse
from timeit import default_timer as timer

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

def inference_edgetpu(runs, image, model, output, label=None):
    # Initialize engine.
    engine = DetectionEngine(model)
    if label:
       labels = ReadLabelFile(label)
    else:
       labels = None
    
    # Open image.
    img = Image.open(image)
    draw = ImageDraw.Draw(img, 'RGBA')
    helvetica=ImageFont.truetype("./Helvetica.ttf", size=72)
    
    initial_h, initial_w = img.size
    frame = img.resize((300, 300))
    
    # Run inference.
    print("Running inferencing for ", runs, " times.")
    
    if runs == 1:
       start = timer()
       ans = engine.DetectWithImage(frame, threshold=0.05, relative_coord=False, top_k=10)
       end = timer()
       print('Elapsed time is ', ((end - start)/runs)*1000, 'ms' )
    else:
       start = timer()
       print('Initial run, discarding.')
       ans = engine.DetectWithImage(frame, threshold=0.05, relative_coord=False, top_k=10)
       end = timer()
       print('First run time is ', (end - start)*1000, 'ms')
       start = timer()
       for i in range(runs):
          ans = engine.DetectWithImage(frame, threshold=0.05, relative_coord=False, top_k=10)
       
       end = timer()
       print('Elapsed time is ', ((end - start)/runs)*1000, 'ms' )
    
    # Display result
    if ans:
      print("Processing output")
      for obj in ans:
        if obj.score > 0.5:
           if labels:
             print(labels[obj.label_id], 'score = ', obj.score)
           else:
             print ('score = ', obj.score)
           box = obj.bounding_box.flatten().tolist()
           bbox = [0]*4
           bbox[0] = box[0]*(initial_h/300)
           bbox[1] = box[1]*(initial_w/300)
           bbox[2] = box[2]*(initial_h/300)
           bbox[3] = box[3]*(initial_w/300)
           print( 'box = ', box )
           draw_rectangle(draw, bbox, (0,128,128,20), width=5)
           if labels:
             draw.text((bbox[0] + 20, bbox[1] + 20), labels[obj.label_id], fill=(255,255,255,20), font=helvetica)
      img.save(output)
      print ('Saved to ', output)
    else:
      print ('No object detected!')

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
    
    result = inference_edgetpu( args.runs, args.input, args.model, output_file, label_file)

if __name__ == '__main__':
  main()

