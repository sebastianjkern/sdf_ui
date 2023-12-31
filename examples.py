from examples.masked_union_example import masked_union_example
from examples.rotation_example import rotation_example
from examples.voronoi_example import voronoi_example
from examples.text_rendering_example import text_rendering_example

from src.sdf_ui import logger

import logging

import argparse

logger().setLevel(logging.CRITICAL)


examples = {
    "masked_union": masked_union_example,
    "rotation": rotation_example,
    "voronoi": voronoi_example,
    "text": text_rendering_example
}

parser = argparse.ArgumentParser(
    prog="SDF_UI Examples",
    description="Show the results of some sdf_ui examples",
)

parser.add_argument('example', type=str, help=f"Options: {list(examples.keys())}")

args = parser.parse_args()

if args.example in examples:
    examples[args.example]()