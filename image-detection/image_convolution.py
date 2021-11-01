from typing import Iterable
from PIL import Image, ImageDraw


def convolve(image: Image.Image,
             kernel: Iterable[Iterable[float]]) -> Image.Image:
    pixels = image.load()
    result = image.copy()
    draw_result = ImageDraw.Draw(result)
    numbands = len(result.mode)
    k_size = len(kernel)
    offset = (k_size // 2)
    for cen_x in range(offset, image.width - offset):
        for cen_y in range(offset, image.height - offset):
            sum = [0] * numbands
            for k_row in range(k_size):
                for k_col in range(k_size):
                    x = (cen_x - offset) + k_col
                    y = (cen_y - offset) + k_row
                    pixel = pixels[x, y] if numbands > 1 else (pixels[x, y],)
                    for band in range(numbands):
                        sum[band] += (pixel[band] * kernel[k_row][k_col])
            sum = tuple(map(round, sum))
            draw_result.point((cen_x, cen_y), sum)

    return result


kernels = dict()
"""Some well-known kernels"""

# Sobel operator: x
kernels["sobel-x"] = [[-1, 0, 1],
                      [-2, 0, 2],
                      [-1, 0, 1]]

# Sobel operator: y
kernels["sobel-y"] = [[-1, -2, -1],
                      [0, 0, 0],
                      [1, 2, 1]]

# sharpens the image
kernels["high-pass"] = [[+0.0, -0.5, +0.0],
                        [-0.5, +3.0, -0.5],
                        [+0.0, -0.5, +0.0]]

# applies box blur
kernels["box-blur"] = [[1 / 9, 1 / 9, 1 / 9],
                       [1 / 9, 1 / 9, 1 / 9],
                       [1 / 9, 1 / 9, 1 / 9]]

# applies Gaussian blur
kernels["gauss-blur"] = [[1 / 256, 4 / 256,  6 / 256,  4 / 256, 1 / 256],
                         [4 / 256, 16 / 256, 24 / 256, 16 / 256, 4 / 256],
                         [6 / 256, 24 / 256, 36 / 256, 24 / 256, 6 / 256],
                         [4 / 256, 16 / 256, 24 / 256, 16 / 256, 4 / 256],
                         [1 / 256, 4 / 256,  6 / 256,  4 / 256, 1 / 256]]


if __name__ == "__main__":

    image = Image.open("sample1.png")
    result = convolve(image.convert("L"), kernels["high-pass"])
    result.show()
