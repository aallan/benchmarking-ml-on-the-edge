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
    print("DetectionEngine support present.")
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
                try:
                    import tensorflow.contrib.tensorrt as trt
                    print("TensorRT support present.")
                    PLATFORM = NVIDIA
                    from object_detection.protos import pipeline_pb2
                    from object_detection.protos import image_resizer_pb2
                    from object_detection import exporter
                    from google.protobuf import text_format
                except:
                    print("No TensorRT support found.")
                    print("Unknown TensorFlow platform.")
                    PLATFORM = 'unknown'
            else:
                print("No GPU support in TensorFlow.")
        except ImportError:
            print("No TensorFlow support found.")

LEGAL_PLATFORMS = NVIDIA
assert PLATFORM in LEGAL_PLATFORMS, "Don't understand platform %s." % PLATFORM

INPUT_NAME='image_tensor'
BOXES_NAME='detection_boxes'
CLASSES_NAME='detection_classes'
SCORES_NAME='detection_scores'
MASKS_NAME='detection_masks'
NUM_DETECTIONS_NAME='num_detections'
FROZEN_GRAPH_NAME='frozen_inference_graph.pb'
PIPELINE_CONFIG_NAME='pipeline.config'
CHECKPOINT_PREFIX='model.ckpt'

import sys
import os
import logging as log
import argparse
import subprocess
from timeit import default_timer as timer

import cv2

from PIL import Image
from PIL import ImageFont, ImageDraw

def make_const6(const6_name='const6'):
    graph = tf.Graph()
    with graph.as_default():
        tf_6 = tf.constant(dtype=tf.float32, value=6.0, name=const6_name)
    return graph.as_graph_def()


def make_relu6(output_name, input_name, const6_name='const6'):
    graph = tf.Graph()
    with graph.as_default():
        tf_x = tf.placeholder(tf.float32, [10, 10], name=input_name)
        tf_6 = tf.constant(dtype=tf.float32, value=6.0, name=const6_name)
        with tf.name_scope(output_name):
            tf_y1 = tf.nn.relu(tf_x, name='relu1')
            tf_y2 = tf.nn.relu(tf.subtract(tf_x, tf_6, name='sub1'), name='relu2')

            #tf_y = tf.nn.relu(tf.subtract(tf_6, tf.nn.relu(tf_x, name='relu1'), name='sub'), name='relu2')
        #tf_y = tf.subtract(tf_6, tf_y, name=output_name)
        tf_y = tf.subtract(tf_y1, tf_y2, name=output_name)
        
    graph_def = graph.as_graph_def()
    graph_def.node[-1].name = output_name

    # remove unused nodes
    for node in graph_def.node:
        if node.name == input_name:
            graph_def.node.remove(node)
    for node in graph_def.node:
        if node.name == const6_name:
            graph_def.node.remove(node)
    for node in graph_def.node:
        if node.op == '_Neg':
            node.op = 'Neg'
            
    return graph_def


def convert_relu6(graph_def, const6_name='const6'):
    # add constant 6
    has_const6 = False
    for node in graph_def.node:
        if node.name == const6_name:
            has_const6 = True
    if not has_const6:
        const6_graph_def = make_const6(const6_name=const6_name)
        graph_def.node.extend(const6_graph_def.node)
        
    for node in graph_def.node:
        if node.op == 'Relu6':
            input_name = node.input[0]
            output_name = node.name
            relu6_graph_def = make_relu6(output_name, input_name, const6_name=const6_name)
            graph_def.node.remove(node)
            graph_def.node.extend(relu6_graph_def.node)
            
    return graph_def


def remove_node(graph_def, node):
    for n in graph_def.node:
        if node.name in n.input:
            n.input.remove(node.name)
        ctrl_name = '^' + node.name
        if ctrl_name in n.input:
            n.input.remove(ctrl_name)
    graph_def.node.remove(node)


def remove_op(graph_def, op_name):
    matches = [node for node in graph_def.node if node.op == op_name]
    for match in matches:
        remove_node(graph_def, match)


def f_force_nms_cpu(frozen_graph):
    for node in frozen_graph.node:
        if 'NonMaxSuppression' in node.name:
            node.device = '/device:CPU:0'
    return frozen_graph


def f_replace_relu6(frozen_graph):
    return convert_relu6(frozen_graph)


def f_remove_assert(frozen_graph):
    remove_op(frozen_graph, 'Assert')
    return frozen_graph

def build_detection_graph(config, checkpoint,
        batch_size=1,
        score_threshold=None,
        force_nms_cpu=True,
        replace_relu6=True,
        remove_assert=True,
        input_shape=None,
        output_dir='.generated_model'):
    """Builds a frozen graph for a pre-trained object detection model"""
    
    config_path = config
    checkpoint_path = checkpoint

    # parse config from file
    config = pipeline_pb2.TrainEvalPipelineConfig()
    with open(config_path, 'r') as f:
        text_format.Merge(f.read(), config, allow_unknown_extension=True)

    # override some config parameters
    if config.model.HasField('ssd'):
        config.model.ssd.feature_extractor.override_base_feature_extractor_hyperparams = True
        if score_threshold is not None:
            config.model.ssd.post_processing.batch_non_max_suppression.score_threshold = score_threshold    
        if input_shape is not None:
            config.model.ssd.image_resizer.fixed_shape_resizer.height = input_shape[0]
            config.model.ssd.image_resizer.fixed_shape_resizer.width = input_shape[1]
    elif config.model.HasField('faster_rcnn'):
        if score_threshold is not None:
            config.model.faster_rcnn.second_stage_post_processing.score_threshold = score_threshold
        if input_shape is not None:
            config.model.faster_rcnn.image_resizer.fixed_shape_resizer.height = input_shape[0]
            config.model.faster_rcnn.image_resizer.fixed_shape_resizer.width = input_shape[1]

    if os.path.isdir(output_dir):
        subprocess.call(['rm', '-rf', output_dir])

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    # export inference graph to file (initial)
    with tf.Session(config=tf_config) as tf_sess:
        with tf.Graph().as_default() as tf_graph:
            exporter.export_inference_graph(
                'image_tensor', 
                config, 
                checkpoint_path, 
                output_dir, 
                input_shape=[batch_size, None, None, 3]
            )

    # read frozen graph from file
    frozen_graph = tf.GraphDef()
    with open(os.path.join(output_dir, FROZEN_GRAPH_NAME), 'rb') as f:
        frozen_graph.ParseFromString(f.read())

    # apply graph modifications
    if force_nms_cpu:
        frozen_graph = f_force_nms_cpu(frozen_graph)
    if replace_relu6:
        frozen_graph = f_replace_relu6(frozen_graph)
    if remove_assert:
        frozen_graph = f_remove_assert(frozen_graph)

    # get input names
    # TODO: handle mask_rcnn 
    input_names = [INPUT_NAME]
    output_names = [BOXES_NAME, CLASSES_NAME, SCORES_NAME, NUM_DETECTIONS_NAME]

    # remove temporary directory
    subprocess.call(['rm', '-rf', output_dir])

    return frozen_graph, input_names, output_names


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
   
   config_path = os.path.join(model, PIPELINE_CONFIG_NAME)
   checkpoint_path = os.path.join(model, CHECKPOINT_PREFIX)
   frozen_graph, input_names, output_names = build_detection_graph(
       config=config_path, checkpoint=checkpoint_path, score_threshold=0.3, batch_size=1)
        
   trt_graph = trt.create_inference_graph(
       input_graph_def=frozen_graph, outputs=output_names, max_batch_size=1,
       max_workspace_size_bytes=1 << 25, precision_mode='FP16', minimum_segment_size=50 )
   
   with tf.Session(config=tf_config) as sess:
        sess.graph.as_default()
        tf.import_graph_def(trt_graph, name='')
        
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
    parser.add_argument('--model', help='Path of the detection model directory.', required=True)
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

