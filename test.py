from src.sdf_ui import *
context = Context((512, 512))
tex = context.texture_from_image("image3.png")
show_texture(tex)