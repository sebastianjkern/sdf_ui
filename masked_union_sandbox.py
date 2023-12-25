import logging

from framework import disc, Context, logger

logger().setLevel(logging.INFO)

with Context((512, 512)) as context:
    disc1 = disc((10, 10), 50)
    disc2 = disc((50, 50), 50)

    sdf1, mask = disc1.masked_union(disc2)

    sdf1.show()
    mask.show()