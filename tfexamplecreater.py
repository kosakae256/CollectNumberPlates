import tensorflow as tf
from object_detection.utils import dataset_util
import contextlib2
import random

def create_tf_example(height,
                      width,
                      filename,
                      image_format,
                      xmins,xmaxs,
                      ymins,
                      ymaxs,
                      classes_text,
                      classes):
  # TODO(user): Populate the following variables from your example.
  # height = None # Image height
  # width = None # Image width
  # filename = None # Filename of the image. Empty if image is not from file
  # encoded_image_data = None # Encoded image bytes
  # image_format = None # b'jpeg' or b'png'

  # xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
  # xmaxs = [] # List of normalized right x coordinates in bounding box
  #            # (1 per box)
  # ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
  # ymaxs = [] # List of normalized bottom y coordinates in bounding box
  #            # (1 per box)
  # classes_text = [] # List of string class name of bounding box (1 per box)
  # classes = [] # List of integer class id of bounding box (1 per box)

  with tf.io.gfile.GFile(f"C:/Users/kosakae256/Documents/Kosakae-Deployment/CollectNumberPlates/formimgs/form{filename}", 'rb') as fid:
      encoded_jpg = fid.read()
      # encoded_jpg_io = io.BytesIO(encoded_jpg)

  tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename.encode("utf-8")),
      'image/source_id': dataset_util.bytes_feature(filename.encode("utf-8")),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
  }))
  return tf_example

from object_detection.dataset_tools import tf_record_creation_util
def output_tfrecord(tfdata,datale):


    output_path=f'C:/Users/kosakae256/Documents/Kosakae-Deployment/CollectNumberPlates/tfrecords/{str(datale+200).zfill(6)}.tfrecord'

    writer = tf.io.TFRecordWriter(output_path)

    height = tfdata["height"]
    width = tfdata["width"]
    filename = tfdata["filename"]

    image_format = b"jpg"

    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    for i in range(0,len(tfdata["xmins"])):
        xmins.append(tfdata["xmins"][i] / width)
        xmaxs.append(tfdata["xmaxs"][i] / width)
        ymins.append(tfdata["ymins"][i] / height)
        ymaxs.append(tfdata["ymaxs"][i] / height)

    classes_text = []
    for text in tfdata["classes_text"]:
        classes_text.append(text.encode("utf-8"))

    classes = []
    for label in tfdata["classes"]:
        classes.append(label)

    tf_example = create_tf_example(height,width,filename,image_format,xmins,xmaxs,ymins,ymaxs,classes_text,classes)
    writer.write(tf_example.SerializeToString())
