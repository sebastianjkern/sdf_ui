from sdf_ui import Canvas, color, sdf


def render_api_example():
    with Canvas((640, 360)) as ctx:
        base = color.clear("#ffffff").cache("base")
        blob = (
            sdf.circle(("35%x", "50%y"), "20%min")
            .smooth_union(sdf.circle(("55%x", "50%y"), "20%min"), k="2%min")
            .fill("#62bb47", (0.0, 0.0, 0.0, 0.0))
            .cache("blob")
        )
        outline = (
            sdf.rect(("50%x", "50%y"), ("55%x", "35%y"), ("5%min", "5%min", "5%min", "5%min"))
            .outline("#009ddc", inflate=2.0)
            .transparency(0.6)
            .cache("outline")
        )

        base_scene = base.alpha_overlay(blob).cache("base_scene")
        framed_scene = base_scene.alpha_overlay(outline).uncached()
        cache = {}

        # Reuse one cache across both renders so the shared base scene only runs once.
        ctx.save(base_scene, "out/render_api_base.png", cache=cache)
        ctx.save(framed_scene, "out/render_api_framed.png", cache=cache)
        framed_scene.show(canvas=ctx, cache=cache)
