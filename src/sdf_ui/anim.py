__docformat__ = "google"

"""Rudimentary video export support. images need to be exported beforehand, and are combined using the function down below"""

import cv2

from .log import logger


def convert_to_video(name, image_paths):
    """
    Convert a sequence of images into a video.

    Parameters:
    - name (str): The name of the output video file (without extension).
    - image_paths (list): List of file paths of input images.

    Example:
    >>> images = ["frame1.jpg", "frame2.jpg", "frame3.jpg"]
    >>> convert_to_video("output_video", images)
    """
    frame = cv2.imread(image_paths[0])
    height, width, _ = frame.shape

    video = cv2.VideoWriter(name + ".mp4", 0, 60, (width, height))

    for image in image_paths:
        video.write(cv2.imread(image))

    cv2.destroyAllWindows()
    video.release()
    logger().info("Finished writing video")
