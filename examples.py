from examples.masked_union_example import masked_union_example
from examples.rotation_example import rotation_example
from examples.voronoi_example import voronoi_example
from examples.text_rendering_example import text_rendering_example
from examples.transparency_example import transparency_example
from examples.animation_example import animation_example
from examples.complex_shapes_example import complex_shapes_example
from examples.font_example import font_example
from examples.arc_example import arc_example
from examples.partial_derivative import partial_derivative_example

from src.sdf_ui import logger

import logging

import argparse

logger().setLevel(logging.CRITICAL)


examples = {
    "masked_union": masked_union_example,
    "rotation": rotation_example,
    "voronoi": voronoi_example,
    "text": text_rendering_example,
    "transparency": transparency_example,
    "font": font_example,
    "complex": complex_shapes_example,
    "anim": animation_example,
    "arc": arc_example,
    "pd": partial_derivative_example
}

parser = argparse.ArgumentParser(
    prog="SDF_UI Examples",
    description="Show the results of some sdf_ui examples",
)

parser.add_argument('example', type=str, help=f"Options: {list(examples.keys())}")

args = parser.parse_args()

if args.example in examples:
    examples[args.example]()