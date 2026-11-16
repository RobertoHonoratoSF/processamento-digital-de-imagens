"""Operações de PDI puras: recebem e devolvem matrizes NumPy uint8."""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def _u8(image: np.ndarray) -> np.ndarray:
    return np.clip(image, 0, 255).astype(np.uint8)


def _channels(image: np.ndarray, function) -> np.ndarray:
    if image.ndim == 2:
        return function(image)
    return np.stack([function(image[..., i]) for i in range(3)], axis=-1)


def grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.copy()
    return _u8(
        0.2126 * image[..., 0] + 0.7152 * image[..., 1] + 0.0722 * image[..., 2]
    )


def decompose_rgb(image: np.ndarray) -> dict[str, np.ndarray]:
    if image.ndim != 3:
        raise ValueError("A decomposição RGB exige uma imagem colorida.")
    return {name: image[..., i].copy() for i, name in enumerate(("R", "G", "B"))}


def decompose_hsv(image: np.ndarray) -> dict[str, np.ndarray]:
    if image.ndim != 3:
        raise ValueError("A decomposição HSV exige uma imagem colorida.")
    rgb = image.astype(np.float64) / 255.0
    maximum = rgb.max(axis=2)
    minimum = rgb.min(axis=2)
    delta = maximum - minimum
    hue = np.zeros_like(maximum)
    mask = delta > 0
    red = mask & (maximum == rgb[..., 0])
    green = mask & (maximum == rgb[..., 1])
    blue = mask & (maximum == rgb[..., 2])
    hue[red] = ((rgb[..., 1][red] - rgb[..., 2][red]) / delta[red]) % 6
    hue[green] = (rgb[..., 2][green] - rgb[..., 0][green]) / delta[green] + 2
    hue[blue] = (rgb[..., 0][blue] - rgb[..., 1][blue]) / delta[blue] + 4
    hue /= 6.0
    saturation = np.divide(delta, maximum, out=np.zeros_like(delta), where=maximum > 0)
    return {"H": _u8(hue * 255), "S": _u8(saturation * 255), "V": _u8(maximum * 255)}


def threshold(image: np.ndarray, k: int) -> np.ndarray:
    return np.where(grayscale(image) >= k, 255, 0).astype(np.uint8)


def logarithmic(image: np.ndarray, gain: float) -> np.ndarray:
    normalized = image.astype(np.float64) / 255.0
    return _u8(255 * gain * np.log1p(normalized) / np.log(2.0))


def power_law(image: np.ndarray, gain: float, gamma: float) -> np.ndarray:
    return _u8(255 * gain * (image.astype(np.float64) / 255.0) ** gamma)


def histogram(image: np.ndarray) -> np.ndarray:
    if image.ndim != 2:
        raise ValueError("O histograma de intensidade exige imagem em escala de cinza.")
    return np.bincount(image.ravel(), minlength=256)


def equalize_histogram(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if image.ndim != 2:
        raise ValueError("A equalização exige imagem em escala de cinza.")
    before = histogram(image)
    cdf = before.cumsum()
    nonzero = cdf[cdf > 0]
    if not len(nonzero) or nonzero[0] == cdf[-1]:
        result = image.copy()
    else:
        lut = np.round((cdf - nonzero[0]) * 255 / (cdf[-1] - nonzero[0]))
        result = _u8(lut[image])
    return result, before, histogram(result)


def intensity_slice(image: np.ndarray, low: int, high: int, preserve: bool) -> np.ndarray:
    base = grayscale(image)
    selected = (base >= low) & (base <= high)
    background = base if preserve else np.zeros_like(base)
    return np.where(selected, 255, background).astype(np.uint8)


def gaussian_mean(image: np.ndarray, size: int, sigma: float) -> np.ndarray:
    radius = (size - 1) / 2
    truncate = radius / sigma
    return _channels(image, lambda c: _u8(ndimage.gaussian_filter(c, sigma, mode="reflect", truncate=truncate)))


def order_filter(image: np.ndarray, size: int, kind: str) -> np.ndarray:
    filters = {
        "Mediana": ndimage.median_filter,
        "Mínimo": ndimage.minimum_filter,
        "Máximo": ndimage.maximum_filter,
    }
    if kind not in filters:
        raise ValueError(f"Filtro de ordem desconhecido: {kind}")
    return _channels(image, lambda c: _u8(filters[kind](c, size=size, mode="reflect")))


def adaptive_median(image: np.ndarray, max_size: int) -> np.ndarray:
    """Filtro adaptativo de mediana clássico, com janelas ímpares crescentes."""
    def channel(data: np.ndarray) -> np.ndarray:
        medians = {s: ndimage.median_filter(data, s, mode="reflect") for s in range(3, max_size + 1, 2)}
        minima = {s: ndimage.minimum_filter(data, s, mode="reflect") for s in medians}
        maxima = {s: ndimage.maximum_filter(data, s, mode="reflect") for s in medians}
        result = data.copy()
        unresolved = np.ones(data.shape, dtype=bool)
        for size in medians:
            med, low, high = medians[size], minima[size], maxima[size]
            valid_median = (med > low) & (med < high) & unresolved
            noisy_pixel = (data <= low) | (data >= high)
            result[valid_median & noisy_pixel] = med[valid_median & noisy_pixel]
            unresolved[valid_median] = False
        result[unresolved] = medians[max_size][unresolved]
        return result
    return _channels(image, channel)


def unsharp_mask(image: np.ndarray, size: int, sigma: float, gain: float) -> np.ndarray:
    blurred = gaussian_mean(image, size, sigma).astype(np.float64)
    return _u8(image.astype(np.float64) + gain * (image.astype(np.float64) - blurred))


def laplacian_enhance(image: np.ndarray, size: int, gain: float) -> tuple[np.ndarray, np.ndarray]:
    """Realce Laplaciano; size>3 faz pré-suavização na janela solicitada."""
    def channel(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        source = data.astype(float)
        if size > 3:
            source = ndimage.uniform_filter(source, size=size, mode="reflect")
        lap = ndimage.laplace(source, mode="reflect")
        enhanced = _u8(data.astype(float) - gain * lap)
        visual = _u8(127.5 + lap / (np.max(np.abs(lap)) or 1) * 127.5)
        return enhanced, visual
    if image.ndim == 2:
        return channel(image)
    pairs = [channel(image[..., i]) for i in range(3)]
    return np.stack([p[0] for p in pairs], -1), np.stack([p[1] for p in pairs], -1)


def sobel_gradient(image: np.ndarray, size: int) -> np.ndarray:
    """Magnitude de Sobel; size>3 controla uma pré-suavização local."""
    def channel(data: np.ndarray) -> np.ndarray:
        source = data.astype(float)
        if size > 3:
            source = ndimage.uniform_filter(source, size=size, mode="reflect")
        gx = ndimage.sobel(source, axis=1, mode="reflect")
        gy = ndimage.sobel(source, axis=0, mode="reflect")
        magnitude = np.hypot(gx, gy)
        return _u8(magnitude / (magnitude.max() or 1) * 255)
    return _channels(image, channel)


def frequency_filter(
    image: np.ndarray, family: str, pass_type: str, cutoff: float, order: int = 2
) -> np.ndarray:
    rows, cols = image.shape[:2]
    y, x = np.ogrid[:rows, :cols]
    distance = np.sqrt((y - rows / 2) ** 2 + (x - cols / 2) ** 2)
    if family == "Gaussiano":
        low = np.exp(-(distance**2) / (2 * cutoff**2))
    elif family == "Butterworth":
        low = 1 / (1 + (distance / cutoff) ** (2 * order))
    else:
        raise ValueError(f"Família desconhecida: {family}")
    transfer = low if pass_type == "Passa-baixa" else 1 - low
    def channel(data: np.ndarray) -> np.ndarray:
        spectrum = np.fft.fftshift(np.fft.fft2(data))
        restored = np.real(np.fft.ifft2(np.fft.ifftshift(spectrum * transfer)))
        if pass_type == "Passa-alta":
            restored += 127.5
        return _u8(restored)
    return _channels(image, channel)


def gaussian_noise(
    image: np.ndarray, mean: float, sigma: float, seed: int | None = None
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(mean, sigma, image.shape)
    return _u8(image.astype(float) + noise)


def impulse_noise(
    image: np.ndarray,
    kind: str,
    amount: float,
    salt_ratio: float = 0.5,
    seed: int | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    result = image.copy()
    selector = rng.random(image.shape[:2])
    if kind == "Sal":
        salt, pepper = selector < amount, np.zeros(selector.shape, bool)
    elif kind == "Pimenta":
        pepper, salt = selector < amount, np.zeros(selector.shape, bool)
    elif kind == "Sal e pimenta":
        salt = selector < amount * salt_ratio
        pepper = (selector >= amount * salt_ratio) & (selector < amount)
    else:
        raise ValueError(f"Ruído desconhecido: {kind}")
    result[salt] = 255
    result[pepper] = 0
    return result
