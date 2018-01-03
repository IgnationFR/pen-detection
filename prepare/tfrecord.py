from typing import List, NamedTuple
from object_detection.core.standard_fields import TfExampleFields
from object_detection.utils import dataset_util
import tensorflow as tf
import tensorflow
import pandas as pd
import os
import random


class _Image(NamedTuple):
    """
    Inherited class from NamedTuple, used to store all info about a image (height, width, annotations...)
    """
    filename: str
    info: pd.DataFrame


def _class_text_to_int(row_label: str):
    if row_label == "pen":
        return 1


def _tf_format(image: _Image, path_to_images: str) -> tensorflow.train.Example:
    """ Transform a image with all its annotations and info to a tensorflow format.

    Args:
        image: All the information about the images (filename, size, annotation...)

    Returns:
        tensorflow.train.Example: The tensorflow object to represent an image and its annotation

    """
    with open(os.path.join(path_to_images, image.filename), "rb") as fp:
        encoded_jpg = fp.read()

    filename = image.filename.encode("utf8")
    image_format = b"jpg"
    xmins, xmaxs, ymins, ymaxs = [], [], [], []
    classes_text = []
    classes = []
    height, width = 0, 0

    for index, row in image.info.iterrows():
        if height == 0 or width == 0:
            height = row["height"]
            width = row["width"]
        xmins.append(row["xmin"] / width)
        xmaxs.append(row["xmax"] / width)
        ymins.append(row["ymin"] / height)
        ymaxs.append(row["ymax"] / height)
        classes_text.append(row["class"].encode("utf8"))
        classes.append(_class_text_to_int(row["class"]))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        TfExampleFields.height: dataset_util.int64_feature(height),
        TfExampleFields.width: dataset_util.int64_feature(width),
        TfExampleFields.filename: dataset_util.bytes_feature(filename),
        TfExampleFields.source_id: dataset_util.bytes_feature(filename),
        TfExampleFields.image_encoded: dataset_util.bytes_feature(encoded_jpg),
        TfExampleFields.image_format: dataset_util.bytes_feature(image_format),
        TfExampleFields.object_bbox_xmin: dataset_util.float_list_feature(xmins),
        TfExampleFields.object_bbox_xmax: dataset_util.float_list_feature(xmaxs),
        TfExampleFields.object_bbox_ymin: dataset_util.float_list_feature(ymins),
        TfExampleFields.object_bbox_ymax: dataset_util.float_list_feature(ymaxs),
        TfExampleFields.object_class_text: dataset_util.bytes_list_feature(classes_text),
        TfExampleFields.object_class_label: dataset_util.int64_list_feature(classes),
    }))

    return tf_example


def _group_image(df: pd.DataFrame, by: str) -> List[_Image]:
    gb = df.groupby(by)
    return [_Image(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_records(path_to_annotations: str, path_to_tf_records: str, path_to_images: str, train_proportion=1.0):
    """ Create two TFRecords file from images and annotations one for training the another for evaluation

    Args:
        path_to_annotations: Path to csv file containing annotations.
        path_to_tf_records: Path to store the two tfrecord files (train and eval).
        train_proportion: Proportion of images use for training.


    """
    annotations = pd.read_csv(path_to_annotations)

    train_writer = tensorflow.python_io.TFRecordWriter(os.path.join(path_to_tf_records, "train.record"))
    eval_writer = tensorflow.python_io.TFRecordWriter(os.path.join(path_to_tf_records, "eval.record"))

    images = _group_image(annotations, by="filename")
    print("%s images" % len(images))

    random.shuffle(images)
    train_size = int(train_proportion * len(images))

    train_data = images[:train_size]
    eval_data = images[train_size:]
    print("%s for training" % len(train_data))
    print("%s for eval" % len(eval_data))

    for image in train_data:
        tf_format_img = _tf_format(image, path_to_images)
        train_writer.write(tf_format_img.SerializeToString())

    for image in eval_data:
        tf_format_img = _tf_format(image, path_to_images)
        eval_writer.write(tf_format_img.SerializeToString())

    eval_writer.close()
    train_writer.close()


if __name__ == "__main__":
    print(os.getcwd())
    create_tf_records("../dataset/annotations.csv",
                      "../dataset/tf-records",
                      "../dataset/renamed-images",
                      train_proportion=0.8)
