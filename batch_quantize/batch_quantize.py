from typing import List

import numba_kmeans
import numpy as np
from PIL import Image

DITHERMODES = {
    "none": Image.Dither.NONE,
    "ordered": Image.Dither.ORDERED,
    "floydsteinberg": Image.Dither.FLOYDSTEINBERG,
    "rasterize": Image.Dither.RASTERIZE
}

SAMPLINGMODES = {
    "nearest": Image.Resampling.NEAREST,
    "box": Image.Resampling.BOX,
    "bilinear": Image.Resampling.BILINEAR,
    "hamming": Image.Resampling.HAMMING,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS
}

def quantize(
    batch: List[Image.Image],
    colors: int = 16,
    dither: str | None = "floydsteinberg",
    sampling_factor=0.5,
    sampling: str = "bicubic",
    max_iter: int = 100,
    tol: float = 0.25,
    patience: int = 3,
    device: str = "cuda",
) -> tuple[List[Image.Image], List[List[int]]]:
    """
    Applies k-means color quantization to a batch of RGB images, followed by optional dithering.

    Parameters:
        batch (List[Image.Image]): List of PIL Image objects (must all be the same size).
        colors (int): Number of colors to quantize each image to. Default is 16.
        dither (str | None): Dithering mode to apply after quantization. Options are:
            "None", "ordered", "floydsteinberg", "rasterize". Default is "floydsteinberg".
        sampling_factor (float): Rescaling factor used for k-means clustering to reduce
            computational load. Must be in (0, 1]. Default is 0.5.
        sampling (str): Sampling mode for resampling images. Options are:
            "nearest", "box", "bilinear", "hamming", "bicubic", "lanczos". Default is "bicubic"
        max_iter (int): Maximum number of iterations for the k-means algorithm. Default is 100.
        tol (float): Convergence threshold for k-means. Default is 0.25.
        patience (int): Number of iterations with no improvement before early stopping. Default is 3.
        device (str): Device to run the k-means algorithm on. Either "cuda" or "cpu". Default is "cuda".

    Returns:
        Tuple[List[Image.Image], List[List[int]]]: 
            - A list of quantized PIL Image objects in 'P' mode with applied dithering.
            - A list of color palettes used for each image, where each palette is a flat list of RGB integers.

    Raises:
        AssertionError: If the images in the batch are not all the same size.
        AssertionError: If an unsupported device is specified.
    """
    assert len(set([img.size for img in batch])) == 1, "All images in batch must be same size"
    assert device in ("cuda", "cpu"), "Only cuda and cpu are supported"

    width, height = batch[0].size
    if sampling_factor < 1.0:
        width, height = int(width * sampling_factor), int(height * sampling_factor)
        batch = [img.resize((width, height), resample=sampling.lower()) for img in batch]

    array_batch = np.stack([np.asarray(img).reshape(-1, 3) for img in batch])

    kmeans = numba_kmeans.kmeans_cuda_batched if device=="cuda" else numba_kmeans.kmeans_cpu_batched

    centroids, _ = kmeans(array_batch, n_clusters=colors, max_iter=max_iter, tol=tol, patience=patience)

    result_batch = []
    result_palettes = []
    for img, centroid in zip(batch, centroids):
        palette = centroid.flatten().astype(int).tolist()
        placeholder_img = Image.new("P", (1, 1))
        placeholder_img.putpalette(palette)
        img = img.quantize(colors, palette=placeholder_img, dither=DITHERMODES[str(dither).lower()])
        result_batch.append(img)
        result_palettes.append(palette)
    
    return result_batch, result_palettes