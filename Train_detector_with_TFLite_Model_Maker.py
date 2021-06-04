#!/usr/bin/env python
# coding: utf-8


# Train a detector with TensorFlow Lite Model Maker



import numpy as np
import os

import tflite_model_maker
from tflite_model_maker.config import ExportFormat
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector


import tensorflow as tf
assert tf.__version__.startswith('2')

tf.get_logger().setLevel('ERROR')
from absl import logging
logging.set_verbosity(logging.ERROR)


# **Step 1. Choose an object detection model archiecture.**
spec2 = tflite_model_maker.object_detector.EfficientDetLite2Spec(
    model_name='efficientdet-lite2',
    uri='https://tfhub.dev/tensorflow/efficientdet/lite2/feature-vector/1', 
    hparams='', model_dir=None, epochs=300, batch_size=64,
    steps_per_execution=1, moving_average_decay=0,
    var_freeze_expr='(efficientnet|fpn_cells|resample_p6)',
    tflite_max_detections=25, strategy=None, tpu=None, gcp_project=None,
    tpu_zone=None, use_xla=False, profile=False, debug=False, tf_random_seed=111111,
    verbose=0)
print("Spec read")


# **Step 2. Load the dataset.**
train_data = object_detector.DataLoader(
    '/home/marco/carla_training/train/annotations_dir/train.record', 
    size=820, 
    label_map={1:'vehicle', 2: 'bike', 3: 'motobike', 4: 'traffic_light', 5: 'traffic_sign'}, 
    annotations_json_file=None
)
validation_data = object_detector.DataLoader(
    '/home/marco/carla_training/test/annotations_dir/val.record', 
    size=208, 
    label_map={1:'vehicle', 2: 'bike', 3: 'motobike', 4: 'traffic_light', 5: 'traffic_sign'}, 
    annotations_json_file=None
)
print("Done!")


#!pip uninstall numpy -y
#!pip install update numpy==1.17.4
import numpy
numpy.__version__


# **Step 3. Train the TensorFlow model with the training data.**
model = object_detector.create(train_data, model_spec=spec2, batch_size=8, train_whole_model=True, validation_data=validation_data, do_train=True)


# **Step 4. Evaluate the model with the test data.**
model.evaluate(validation_data)


# **Step 5.  Export as a TensorFlow Lite model.**
model.export(export_dir='model2.tflite')


# **Step 6.  Evaluate the TensorFlow Lite model.**
model.evaluate_tflite('model2.tflite', validation_data)


# **Step 7.  Run object detection and show the detection results

#@title Load the trained TFLite model and define some visualization functions

import cv2

from PIL import Image

# Load the labels into a list
classes = ['???'] * model.model_spec.config.num_classes
label_map = model.model_spec.config.label_map
for label_id, label_name in label_map.as_dict().items():
  classes[label_id-1] = label_name

# Define a list of colors for visualization
COLORS = np.random.randint(0, 255, size=(len(classes), 3), dtype=np.uint8)

def preprocess_image(image_path, input_size):
  """Preprocess the input image to feed to the TFLite model"""
  img = tf.io.read_file(image_path)
  img = tf.io.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.uint8)
  original_image = img
  resized_img = tf.image.resize(img, input_size)
  resized_img = resized_img[tf.newaxis, :]
  return resized_img, original_image


def set_input_tensor(interpreter, image):
  """Set the input tensor."""
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image


def get_output_tensor(interpreter, index):
  """Retur the output tensor at the given index."""
  output_details = interpreter.get_output_details()[index]
  tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
  return tensor


def detect_objects(interpreter, image, threshold):
  """Returns a list of detection results, each a dictionary of object info."""
  # Feed the input image to the model
  set_input_tensor(interpreter, image)
  interpreter.invoke()

  # Get all outputs from the model
  boxes = get_output_tensor(interpreter, 0)
  classes = get_output_tensor(interpreter, 1)
  scores = get_output_tensor(interpreter, 2)
  count = int(get_output_tensor(interpreter, 3))

  results = []
  for i in range(count):
    if scores[i] >= threshold:
      result = {
        'bounding_box': boxes[i],
        'class_id': classes[i],
        'score': scores[i]
      }
      results.append(result)
  return results


def run_odt_and_draw_results(image_path, interpreter, threshold=0.5):
  """Run object detection on the input image and draw the detection results"""
  # Load the input shape required by the model
  _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

  # Load the input image and preprocess it
  preprocessed_image, original_image = preprocess_image(
      image_path, 
      (input_height, input_width)
    )

  # Run object detection on the input image
  results = detect_objects(interpreter, preprocessed_image, threshold=threshold)

  # Plot the detection results on the input image
  original_image_np = original_image.numpy().astype(np.uint8)
  for obj in results:
    # Convert the object bounding box from relative coordinates to absolute 
    # coordinates based on the original image resolution
    ymin, xmin, ymax, xmax = obj['bounding_box']
    xmin = int(xmin * original_image_np.shape[1])
    xmax = int(xmax * original_image_np.shape[1])
    ymin = int(ymin * original_image_np.shape[0])
    ymax = int(ymax * original_image_np.shape[0])

    # Find the class index of the current object
    class_id = int(obj['class_id'])

    # Draw the bounding box and label on the image
    color = [int(c) for c in COLORS[class_id]]
    cv2.rectangle(original_image_np, (xmin, ymin), (xmax, ymax), color, 2)
    # Make adjustments to make the label visible for all objects
    y = ymin - 15 if ymin - 15 > 15 else ymin + 15
    label = "{}: {:.0f}%".format(classes[class_id], obj['score'] * 100)
    cv2.putText(original_image_np, label, (xmin, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

  # Return the final image
  original_uint8 = original_image_np.astype(np.uint8)
  return original_uint8



model_path = 'model2.tflite'
#INPUT_IMAGE_URL = "https://techtime.news/wp-content/uploads/sites/2/2017/10/Cognata-Simulation-Engine.png"
INPUT_IMAGE_URL = "https://carla.readthedocs.io/en/0.9.7/img/low_quality_capture.png"

DETECTION_THRESHOLD = 0.3 #@param {type:"number"}

TEMP_FILE = '/tmp/result.png'

get_ipython().system('wget -q -O $TEMP_FILE $INPUT_IMAGE_URL')
im = Image.open(TEMP_FILE)
im.thumbnail((512, 512), Image.ANTIALIAS)
im.save(TEMP_FILE, 'PNG')

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Run inference and draw detection result on the local copy of the original file
detection_result_image = run_odt_and_draw_results(
    TEMP_FILE, 
    interpreter, 
    threshold=DETECTION_THRESHOLD
)

# Show the detection result
im = Image.fromarray(detection_result_image)
im.thumbnail((512, 512), Image.ANTIALIAS)
im.save("result2.png")




get_ipython().system('edgetpu_compiler --min_runtime_version 13 model.tflite')

