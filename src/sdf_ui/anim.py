# Rudimentary video export support. 
# images need to be exported beforehand, 
# and are combined using the function down below
import cv2

from sdf_ui.core.log import logger


def convert_to_video(name, image_paths):
    frame = cv2.imread(image_paths[0])
    height, width, _ = frame.shape

    video = cv2.VideoWriter(name + ".mp4", 0, 60, (width, height))

    for image in image_paths:
        video.write(cv2.imread(image))

    cv2.destroyAllWindows()
    video.release()
    logger().info("Finished writing video")
