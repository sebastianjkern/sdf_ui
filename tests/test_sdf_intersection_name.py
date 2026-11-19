from sdf_ui import sdf


def test_intersection_is_explicit_canonical_method():
    first = sdf.circle((8, 8), 5)
    second = sdf.circle((10, 8), 5)

    canonical = first.intersection(second)
    alias = first.intersect(second)

    assert canonical.op == alias.op == "intersection"
    assert [node.op for node in canonical.inputs] == ["circle", "circle"]
    assert [node.op for node in alias.inputs] == ["circle", "circle"]
