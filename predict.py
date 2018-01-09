import datetime
import numpy as np
import os
import glob
import tensorflow as tf
from PIL import Image
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


def _to_numpy_array(img: Image) -> np.ndarray:
    """ Transform a PIL.Image in a numpy array.

    Args:
        img: PIL.Image to transform.

    Returns: a numpy array represents the given image.

    """
    (im_width, im_height) = img.size
    return np.array(img.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def predict(images_path: str, results_path: str):
    """ Predict the bounding boxes and classes for each image in given path.
    Draw the result on each image and save them in the second given path.

    Args:
        images_path: Path to images to be predicted.
        results_path: Path where to save the result images.

    """
    inference_graph_path = os.path.join("model", "output_inference_graph", "frozen_inference_graph.pb")
    label_map_path = os.path.join("data", "label_map.pbtxt")
    num_class = 1

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(inference_graph_path, "rb") as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name="")

    label_map = label_map_util.load_labelmap(label_map_path)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_class,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    test_images = glob.glob(images_path + "*.jpg")

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            image_tensor = detection_graph.get_tensor_by_name("image_tensor:0")
            tensors = [
                image_tensor,
                detection_graph.get_tensor_by_name("detection_boxes:0"),
                detection_graph.get_tensor_by_name("detection_scores:0"),
                detection_graph.get_tensor_by_name("detection_classes:0"),
                detection_graph.get_tensor_by_name("num_detections:0"),
            ]

            for image_path in test_images:
                image = Image.open(image_path)
                image_np = _to_numpy_array(image)
                image_np_expanded = np.expand_dims(image_np, axis=0)

                start = datetime.datetime.now()

                (boxes, scores, classes, num) = sess.run(
                    fetches=tensors,
                    feed_dict={image_tensor: image_np_expanded})
                print("took %s ms" % (start - datetime.datetime.now()).microseconds)

                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8)
                im = Image.fromarray(image_np)
                im.save(os.path.join(results_path, image_path.split("/")[-1]))


if __name__ == "__main__":
    predict("data/test", "model/test")