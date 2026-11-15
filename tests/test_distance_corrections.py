import numpy as np
import pytest

from sdf_ui import sdf
from sdf_ui.core.sdf import SDFTexture
from sdf_ui.gpu.nodes._distance import normalize_distance, redistance


def test_local_normalization_corrects_gradient_at_zero_contour():
    yy, xx = np.mgrid[-8:9, -8:9].astype(np.float32)
    field = 0.5 * xx

    corrected = normalize_distance(field, 1e-3, 32.0)
    grad_y, grad_x = np.gradient(corrected)
    magnitude = np.hypot(grad_x, grad_y)

    np.testing.assert_allclose(corrected[:, 8], 0.0, atol=1e-7)
    np.testing.assert_allclose(magnitude[4:-4, 8], 1.0, atol=1e-3)


def test_local_normalization_rejects_invalid_epsilon():
    with pytest.raises(ValueError, match="epsilon"):
        normalize_distance(np.zeros((4, 4), dtype=np.float32), 0, 32.0)


def test_local_normalization_does_not_amplify_flat_regions():
    field = np.full((5, 5), 0.25, dtype=np.float32)

    np.testing.assert_array_equal(normalize_distance(field, 1e-3, 32.0), field)


def test_local_normalization_fades_out_away_from_the_surface():
    yy, xx = np.mgrid[0:9, 0:17].astype(np.float32)
    field = xx * 4.0

    corrected = normalize_distance(field, 1e-3, 8.0)

    np.testing.assert_allclose(corrected[:, 0], 0.0, atol=1e-7)
    np.testing.assert_array_equal(corrected[:, -1], field[:, -1])


def test_redistance_rebuilds_subpixel_signed_distance_to_a_line():
    yy, xx = np.mgrid[0:17, 0:21].astype(np.float32)
    field = 0.2 * (xx - 10.25)

    result = redistance(field)

    np.testing.assert_allclose(result, xx - 10.25, atol=1e-6)
    assert result[8, 2] < 0 < result[8, 18]


def test_redistance_rejects_a_field_without_a_zero_contour():
    with pytest.raises(ValueError, match="zero contour"):
        redistance(np.ones((8, 8), dtype=np.float32))


def test_distance_corrections_are_exposed_as_sdf_methods():
    shape = sdf.circle((20, 20), 12).smooth_union(sdf.circle((30, 20), 12), k=5)

    assert isinstance(shape.normalize_distance(), SDFTexture)
    assert isinstance(shape.redistance(), SDFTexture)
