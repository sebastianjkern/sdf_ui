---
name: sdf-ui-text-rendering
description: Implement or diagnose lettering in sdf-ui scenes and examples. Use for small or jagged text, glyph coverage and fallback, text placement, and lettering in PNG, SVG, or pen-plotter output.
---

# Render and inspect sdf-ui text

## Shape and place text

- Use the shared text API in `src/sdf_ui/text.py` for SDF scenes. Pass the same font, fallback, shaping, and spacing options to `text` and `measure_text`. Pass the text box's width and height to measurement when checking wrapping or fit.
- Distinguish advance width from visible ink bounds. For SDF text, `ox` and `oy` describe baseline placement and boxes use a lower-left origin. Exported `VectorPath` coordinates use a top-left origin. Convert at the boundary and check accents and descenders after placement.
- Use the bundled font for reproducible examples. If a string needs broader coverage, pass explicit fallback fonts and use `missing_glyph="error"` while checking the actual strings.

## Keep glyph edges through export

- Trace jagged edges back to the first low-resolution sampling or threshold. Enlarging a glyph mask after it has been binarized preserves the staircase. For SDF text, distinguish glyph curve sampling and cached glyph resolution from the final raster resolution. Inspect `samples`, `oversample`, and `min_render_size` when the glyph field itself is coarse. Use a scaled canvas when the target is a higher-resolution raster at the same layout size.
- For PNG lettering built from a font mask, rasterize at a multiple of the target size, keep grayscale coverage, and downsample once with an antialiasing filter. Allow padding around ink bounds for accents and the filter footprint.
- For SVG, prefer filled glyph outlines or contours traced from a supersampled coverage mask. Preserve the holes in letters such as O and A with a suitable fill rule. For G-code, derive pen centerlines at a spacing suited to the physical nib and plot scale. Reuse the same lettering placement across outputs.
- `examples/alpine_engraving.py` demonstrates these export choices with bundled Roboto, 6× text sampling, filled SVG contours, and separate pen strokes for G-code. Choose the sampling scale for the smallest lettering in the current piece.

## Verify the result

Inspect a crop at final PNG size and at higher magnification. Check small labels, diagonal stems, round counters, and any accents or descenders used in the piece. Open the SVG at high zoom and check that the same glyphs retain their shape and placement. For a text-only fix, compare an unaffected crop to the prior render to catch changes to the surrounding artwork.
