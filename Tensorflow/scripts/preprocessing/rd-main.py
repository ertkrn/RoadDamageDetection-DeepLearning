import numpy as np
import sys
import os
import tarfile
import tensorflow as tf
import zipfile
import cv2
import random

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

parentpath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
parentpath = os.path.abspath(os.path.join(parentpath, '..'))

sys.path.append(parentpath + '/models/research/')


from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util


parentdir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
parentdir = os.path.abspath(os.path.join(parentdir, '..'))
parentdir = os.path.abspath(os.path.join(parentdir, '..'))
parentdir = parentdir + '/trainedModels/'

flags = tf.app.flags
flags.DEFINE_string('path_to_image', '',
                    'Path to a image '
                    'file. If provided, other configs are ignored')

FLAGS = flags.FLAGS


PATH_TO_CKPT = parentdir + 'frozen_inference_graph.pb'
#PATH_TO_CKPT = parentdir + 'ilkEgitim/frozen_inference_graph.pb'
PATH_TO_LABELS = parentdir + 'crack_label_map.pbtxt'
NUM_CLASSES = 8

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

parentdir = os.path.abspath(os.path.join(parentdir, '..'))
base_path = parentdir + '/RoadDamageDataset/'
# get images from val.txt
PATH_TO_TEST_IMAGES_DIR = base_path
D_TYPE = ['D00', 'D01', 'D10', 'D11', 'D20','D40', 'D43']
govs = ['Adachi', 'Ichihara', 'Muroran', 'Chiba', 'Sumida', 'Nagakute', 'Numazu']

val_list = []
for gov in govs:
    file = open(PATH_TO_TEST_IMAGES_DIR + gov + '/ImageSets/Main/val.txt', 'r')
    for line in file:
        line = line.rstrip('\n').split('/')[-1]
        val_list.append(line)
    file.close()

print("# of validation images：" + str(len(val_list)))

TEST_IMAGE_PATHS=[]
random.shuffle(val_list)

test_path = "" #input a test image path
if FLAGS.path_to_image:
    test_path = FLAGS.path_to_image
else:
    test_path = "" #input a test image path
TEST_IMAGE_PATHS.append(test_path)
# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    for image_path in TEST_IMAGE_PATHS:
        image = Image.open(image_path)
      # the array based representation of the image will be used later in order to prepare the
      # result image with boxes and labels on it.
        image_np = load_image_into_numpy_array(image)
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
      # Actual detection.
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
      # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            min_score_thresh=0.3,
            use_normalized_coordinates=True,
            line_thickness=8)
        fig=plt.figure(figsize=IMAGE_SIZE)
        plt.axis('off')
        plt.title('The image including ')
        plt.imshow(image_np)
        #fig.savefig('demo.png', bbox_inches='tight')
        fig.savefig('', bbox_inches='tight') #enter a path for save