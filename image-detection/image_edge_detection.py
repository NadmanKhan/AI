from image_convolution import convolve, kernels
from PIL import Image, ImageDraw
from math import floor, pi, sqrt, atan2
import numpy as np


def convert_to_grayscale(image: Image.Image) -> Image.Image:
    if image.mode == "L":
        return image
    
    pixels = image.load()
    intensity = np.zeros((image.height, image.width))

    for x in range(image.width):
        for y in range(image.height):
            intensity[y, x] = round(sum(pixels[x, y]) / len(pixels[x, y]))

    return Image.fromarray(intensity)


def compute_gradient(image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    if image.mode != "L":
        image = image.convert("L")
    mag_x = convolve(image, kernels["sobel-x"]).load()
    mag_y = convolve(image, kernels["sobel-y"]).load()
    k_size = len(kernels["sobel-x"])
    offset = (k_size // 2)
    mag = np.zeros(image.size, dtype=int)
    dir = np.zeros(image.size, dtype=int)
    width, height = image.size
    for x in range(offset, width - offset):
        for y in range(offset, height - offset):
            mag[x, y] = round(sqrt((mag_x[x, y] ** 2) + (mag_y[x, y] ** 2)))
            dir[x, y] = atan2(mag_y[x, y], mag_x[x, y])
    return (mag, dir)

def approximate_gradient(image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    if image.mode != "L":
        image = image.convert("L")
    k_size = len(kernels["sobel-x"])
    offset = (k_size // 2)
    mag = np.zeros(image.size, dtype=int)
    dir = np.zeros(image.size, dtype=int)
    width, height = image.size
    pixels = image.load()
    for x in range(offset, width - offset):
        for y in range(offset, height - offset):
            mx = pixels[x + 1, y] - pixels[x - 1, y]
            my = pixels[x, y + 1] - pixels[x, y - 1]
            mag[x, y] = round(sqrt((mx ** 2) + (my ** 2)))
            dir[x, y] = atan2(my, mx)
    return (mag, dir)


def keep_maximum_gradient(gradient: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray,
                                                                            np.ndarray]:
    mag, dir = gradient[0].copy(), gradient[1].copy()
    width, height = mag.shape
    d = ((1, 0), (1, 1), (0, 1), (-1, 1))
    for x in range(1, width - 1):
        for y in range(1, height - 1):
            angle = dir[x, y] + (0 if dir[x, y] >= 0 else pi)
            index = round(angle / (pi / 4)) % 4
            (dx, dy) = d[index]
            (x1, y1), (x2, y2) = (x + dx, y + dy), (x - dx, y - dy)
            if mag[x, y] < mag[x1, y1] or mag[x, y] < mag[x2, y2]:
                mag[x, y] = 0
    return (mag, dir)


def strong_edge_points(mag: np.ndarray,
                       low_threshold_ratio: float = 0.1,
                       high_threshold_ratio: float = 0.2) -> list[tuple[int, int]]:

    if not (0 < low_threshold_ratio < high_threshold_ratio < 1):
        raise ValueError

    (width, height) = mag.shape
    mag_max = int(mag.max())
    low_threshold = round(mag_max * low_threshold_ratio)
    high_threshold = round(mag_max * high_threshold_ratio)
    
    strong_points = set()
    for x in range(width):
        for y in range(height):
            if mag[x, y] >= high_threshold:
                strong_points.add((x, y))

    d = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))

    last_added = strong_points
    while last_added:
        new_points = set()
        for (x, y) in last_added:
            for (dx, dy) in d:
                nx, ny = (x + dx), (y + dy)
                if not (0 <= nx < width and 0 <= ny < height):
                    continue
                if mag[nx, ny] >= low_threshold and (nx, ny) not in strong_points:
                    new_points.add((nx, ny))
        strong_points.update(new_points)
        last_added = new_points

    return list(strong_points)


def detect_edge_points_canny(image: Image.Image) -> list[tuple[int]]:
    grayscaled = image if image.mode == "L" else image.convert("L")
    blurred = convolve(grayscaled, kernels["gauss-blur"])
    gradient = approximate_gradient(blurred)
    gradient = keep_maximum_gradient(gradient)
    points = strong_edge_points(gradient[0])
    return points


def convolve_sobel(image: Image.Image) -> Image.Image:
    mag = compute_gradient(image)[0]
    k_size = len(kernels["sobel-x"])
    offset = (k_size // 2)
    result = Image.new("L", image.size)
    draw_result = ImageDraw.Draw(result)

    for x in range(offset, image.width - offset):
        for y in range(offset, image.height - offset):
            draw_result.point((x, y), int(mag[x, y]))

    return result


if __name__ == "__main__":

    image = Image.open("sample3.png")
    result = Image.new("L", image.size)
    draw = ImageDraw.Draw(result)
    points = detect_edge_points_canny(image)
    for p in points:
        draw.point(p, fill=255)
    result.show()
