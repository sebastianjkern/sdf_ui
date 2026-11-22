from build_tools.lint.repeated_materialization import lint_file


def test_repeated_materialization_reports_loop_comprehension_and_sort_key(tmp_path):
    path = tmp_path / "geometry.py"
    path.write_text(
        "def stitch(points, indices):\n"
        "    for i in indices:\n"
        "        print(tuple(points)[i])\n"
        "    ordered = sorted(indices, key=lambda i: list(points)[i])\n"
        "    return [list(points)[i] for i in ordered]\n"
    )
    assert sorted(line.split(":")[1] for line in lint_file(path)) == ["3", "4", "5"]


def test_repeated_materialization_allows_one_snapshot_and_per_item_conversion(tmp_path):
    path = tmp_path / "geometry.py"
    path.write_text(
        "def stitch(points, groups, i):\n"
        "    point = list(points)[i]\n"
        "    points = tuple(points)\n"
        "    return [tuple(group)[i] for group in groups], points[i], point\n"
    )
    assert list(lint_file(path)) == []
