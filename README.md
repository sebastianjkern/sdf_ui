# SDF UI

What started [here](https://github.com/sebastianjkern/metaballs), has become a not yet fully featured but relatively complex 2d graphics rendering engine with gpu support.

SDF UI is the result of the experiment if you can extend the idea of MSDF text rendering and SDF based circle rendering to all standard 2D rendering stuff. Why don't draw rectangles, bezier curves, lines etc. with SDFs aswell. While in theory this is quiet slow, it offers great features like morphing between figures pretty much out of the box. And the determining factor of rendering cost is the size of the canvas and not the complexity of the object. For example the grid in the picture down below has pretty much the same cost as the rounded rectangle. And pretty much everything can be drawn with antialiasing without additional cost. And depending on what you're drawing, the rendered textures can be used again. 

Many SDFs used by this project were originally created by [Inigo Quilez](https://iquilezles.org/articles/distfunctions2d/).

### Target Specs: 
Unclear what the target for this library is. Somwhere along the line of a educational library for graphics programming, with some esoteric rendering pipeline, but also high fidelity graphics. Maybe usable for high fidelity graphics of mathematical topics? Maybe as plug in replacement for matplotlib rendering backend?

### How it works:

<img src="./sdf_ui_diagram.png" width="500">

### Examples:
___

<img src="./image1.png" width="500" >

Text/Letter rendering:

<img src="./image2.png" width="500" >

Print images to console:

<img src="./console.png" width="500" > 

Freeform Gradient:

<img src="./image3.png" width="500" >

## Ideas for further development

- [ ] Native Support for animation
- [ ] Vector Fields
- [ ] Voronoi sdf (Basically many overlayed circle sdfs)
- [ ] Text rendering (Stroke rendering of single letters already possible)
- [ ] Rendering of text with custom sdf font
- [x] Freeform Gradients (Overlay of Different gradients, possible trough half transparent fills of sdfs)
- [ ] Ortographic Projection of 3D SDFs
- [ ] Fill with images (trough moderngl texture api)
- [ ] YAML based render script, or node based editor, unsure about the alignment with the target specs
