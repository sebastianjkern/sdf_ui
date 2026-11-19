from sdf_ui import sdf


def test_xor_operator_builds_symmetric_difference():
    left = sdf.circle((10, 10), 6)
    right = sdf.circle((14, 10), 6)

    result = left ^ right

    assert result.op == "subtract"
    assert result.inputs[0].op == "union"
    assert result.inputs[1].op == "intersection"


def test_xor_method_matches_operator():
    left = sdf.circle((10, 10), 6)
    right = sdf.circle((14, 10), 6)

    operator_result = left ^ right
    method_result = left.xor(right)
    assert operator_result.op == method_result.op == "subtract"
    assert [node.op for node in operator_result.inputs] == [
        node.op for node in method_result.inputs
    ] == ["union", "intersection"]
