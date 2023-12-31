from src.sdf_ui import *
context = Context((512, 512))
tex = context.texture_from_image("voronoi.png")
context.show_texture(tex)