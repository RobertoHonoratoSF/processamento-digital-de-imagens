import numpy as np
import pytest

from pdi import operations as op


@pytest.fixture
def gray():
    return np.arange(64, dtype=np.uint8).reshape(8, 8) * 4


@pytest.fixture
def rgb(gray):
    return np.stack((gray, np.flipud(gray), np.fliplr(gray)), axis=-1)


def assert_image(result, shape):
    assert result.shape == shape
    assert result.dtype == np.uint8
    assert result.min() >= 0 and result.max() <= 255


def test_color_decompositions(rgb):
    assert set(op.decompose_rgb(rgb)) == {"R", "G", "B"}
    hsv = op.decompose_hsv(rgb)
    assert set(hsv) == {"H", "S", "V"}
    assert all(channel.shape == rgb.shape[:2] for channel in hsv.values())


def test_intensity_operations(gray, rgb):
    for result in (
        op.threshold(rgb, 100), op.logarithmic(gray, 1),
        op.power_law(gray, 1, 2), op.intensity_slice(gray, 50, 150, True),
    ):
        assert result.dtype == np.uint8
    assert set(np.unique(op.threshold(rgb, 100))) <= {0, 255}


def test_equalization_and_histograms(gray):
    result, before, after = op.equalize_histogram(gray)
    assert_image(result, gray.shape)
    assert before.sum() == gray.size == after.sum()


@pytest.mark.parametrize("kind", ["Mediana", "Mínimo", "Máximo"])
def test_spatial_filters(gray, kind):
    assert_image(op.gaussian_mean(gray, 5, 1.2), gray.shape)
    assert_image(op.order_filter(gray, 3, kind), gray.shape)
    assert_image(op.adaptive_median(gray, 5), gray.shape)
    assert_image(op.unsharp_mask(gray, 3, 1, 1), gray.shape)
    enhanced, laplacian = op.laplacian_enhance(gray, 5, 1)
    assert_image(enhanced, gray.shape)
    assert_image(laplacian, gray.shape)
    assert_image(op.sobel_gradient(gray, 5), gray.shape)


@pytest.mark.parametrize("family", ["Gaussiano", "Butterworth"])
@pytest.mark.parametrize("pass_type", ["Passa-baixa", "Passa-alta"])
def test_frequency_filters(rgb, family, pass_type):
    assert_image(op.frequency_filter(rgb, family, pass_type, 4, 2), rgb.shape)


def test_noise_is_reproducible(gray):
    first = op.gaussian_noise(gray, 0, 10, seed=7)
    second = op.gaussian_noise(gray, 0, 10, seed=7)
    assert np.array_equal(first, second)
    impulse = op.impulse_noise(gray, "Sal e pimenta", 0.5, 0.3, seed=7)
    assert_image(impulse, gray.shape)
    assert np.any(impulse == 0) and np.any(impulse == 255)


def test_rejects_incompatible_equalization(rgb):
    with pytest.raises(ValueError):
        op.equalize_histogram(rgb)

