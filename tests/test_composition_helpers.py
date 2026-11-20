
import pytest

from sdf_ui import color, sdf
from sdf_ui.core.expressions import evaluate_expr, param
from sdf_ui.graph import lower_graph
from sdf_ui.learning import Parameter

_WIDTH_TEXT = 'width'

def test_union_all_composes_any_nonempty_iterable():
    first = sdf.circle((4, 5), 3)
    second = sdf.rect((12, 5), (2, 4))

    result = sdf.union_all(shape for shape in (first, second))

    assert result.op == "union"
    assert result.inputs == (first, second)
    assert sdf.union_all((first,)) is first
    with pytest.raises(ValueError, match="at least one shape"):
        sdf.union_all(())
    with pytest.raises(TypeError, match="SDFTexture instances"):
        sdf.union_all((object(),))
    with pytest.raises(TypeError, match="SDFTexture instances"):
        sdf.union_all((first, object()))


def test_over_accepts_ordered_layers_and_noop_call():
    base = color.clear("#101010")
    middle = color.clear("#202020")
    top = color.clear("#303030")

    result = base.over(middle, top)

    assert result.op == "alpha_overlay"
    assert result.inputs[0].op == "alpha_overlay"
    assert result.inputs[0].inputs == (base, middle)
    assert result.inputs[1] is top
    assert base.over() is base


def test_over_keeps_lazy_parameters_in_composed_graphs():
    layer = sdf.circle((8, 8), 4).fill(
        (param("red"), 0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 0.0)
    )
    scene = color.clear("#101010").over(layer, layer)

    plan = lower_graph({"image": scene})

    assert plan.parameter_names == ("red",)


@pytest.mark.parametrize(
    ("factory", "operation"),
    [
        (sdf.rect_from_size, "rect"),
        (sdf.rounded_rect_from_size, "rounded_rect"),
    ],
)
def test_full_size_rectangle_helpers_convert_dimensions_and_keep_lazy_values(
    factory, operation
):
    shape = factory((param("x"), 12), (param(_WIDTH_TEXT), 8), 2)
    params = dict(shape.params)

    assert shape.op == operation
    assert evaluate_expr(params["center"][0], None, {"x": 20}) == 20
    assert evaluate_expr(params["size"][0], None, {_WIDTH_TEXT: 18}) == 9
    assert params["size"][1] == 4
    assert params["corner_radius"] == 2

    plan = lower_graph({"shape": shape})
    assert set(plan.parameter_names) == {_WIDTH_TEXT, "x"}


def test_full_size_rectangles_keep_direct_parameter_leaves_live():
    width = Parameter(_WIDTH_TEXT, 18.0)
    shape = sdf.rounded_rect_from_size((10, 12), (width, 8), 2)

    plan = lower_graph({"shape": shape})

    assert plan.parameter_names == (_WIDTH_TEXT,)
