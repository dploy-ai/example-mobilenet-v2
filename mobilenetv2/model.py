from typing import Dict
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image


class ObjectDetection:
    def __init__(self, model: hub) -> None:
        self.model = model

    def detect(
        self, image: Image, max_n_object: int = 5, min_score: float = 0.15
    ) -> Dict[str, int]:
        """
        :arg
            image: PIL image object
            max_n_object: Integer, set this parameter to get the most
                          probable at most N objects from the image
            min_score: Float, set this parameter between 0 and 1. It means
                       that the object detections having less than  min_score
                       will be discarded.
        """

        img_tensor = self._convert_image_to_tensor(image)
        raw_detections = self.model(img_tensor)
        return self._filter_results(raw_detections, max_n_object, min_score)

    @staticmethod
    def _filter_results(
        raw_detections: dict, max_n_object: int, min_score: float
    ) -> Dict[str, int]:
        """
        :arg
            raw_detections: MobileNet v2 detections
            max_n_object: Integer, set this parameter to get the most probable
                          at most N objects from the image
            min_score: Float, set this parameter between 0 and 1. It means that
                       the object detections having less than min_score
                       will be discarded.
        """
        dict_result = {key: value.numpy() for key, value in raw_detections.items()}
        n_objects = min(
            max_n_object,
            np.where(dict_result["detection_scores"].flatten() >= min_score)[0].size,
        )
        return {
            "detection_class_entities": dict_result["detection_class_entities"][
                0:n_objects
            ],
            "detection_scores": dict_result["detection_scores"][0:n_objects],
            "detection_boxes": dict_result["detection_boxes"][0:n_objects],
            "detection_class_names": dict_result["detection_class_names"][0:n_objects],
            "detection_class_labels": dict_result["detection_class_labels"][
                0:n_objects
            ],
        }

    @staticmethod
    def _convert_image_to_tensor(image: Image) -> tf.Tensor:
        tensor = tf.convert_to_tensor(np.asarray(image))
        image_format = image.format.lower()
        if image_format == "jpeg" or image_format == "jpg":
            tensor = tf.io.encode_jpeg(tensor)
            tensor = tf.image.decode_jpeg(tensor, channels=3)
        elif image_format == "png":
            tensor = tf.image.encode_png(tensor)
            tensor = tf.image.decode_png(tensor, channels=3)
        else:
            raise Exception("We only support jpeg or png formats!")
        return tf.image.convert_image_dtype(tensor, tf.float32)[tf.newaxis, ...]
