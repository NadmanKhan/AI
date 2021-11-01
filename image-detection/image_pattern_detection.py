from collections import defaultdict
from image_edge_detection import detect_edge_points_canny
from math import cos, sin, pi
from PIL import Image, ImageDraw


def pattern_detect(image: Image.Image, radius_range: tuple[int, int] = (17, 21),
                   threshold_ratio: float = 0.5):
    if not radius_range[0] <= radius_range[1]:
        raise ValueError
    if not 0 < threshold_ratio <= 1:
        raise ValueError

    r_min, r_max = radius_range

    total_points = dict()
    for r in range(r_min, r_max + 1):
        total_points[r] = round(2 * pi * r)

    circle_params = []
    for r in range(r_min, r_max + 1):
        total = total_points[r]
        for t in range(total):
            angle = 2 * pi * (t / total)
            dx = round(r * cos(angle))
            dy = round(r * sin(angle))
            circle_params.append((dx, dy, r))

    strong_points = detect_edge_points_canny(image)

    counts = defaultdict(int)
    for (x, y) in strong_points:
        for (dx, dy, r) in circle_params:
            a = x + dx
            b = y + dy
            if 0 <= a < image.width and 0 <= b < image.height:
                counts[(a, b, r)] += 1

    def distant(circle1: tuple[int, int, int], circle2: tuple[int, int, int]) -> bool:
        (a1, b1, r1), (a2, b2, r2) = circle1, circle2
        return ((a1 - a2) ** 2) + ((b1 - b2) ** 2) > (r2 ** 2)

    circle_detect_ratios: list[tuple[tuple[int, int, int], float]] = []
    for ((a, b, r), count) in counts.items():
        ratio = count / total_points[r]
        if (ratio >= threshold_ratio):
            circle_detect_ratios.append(((a, b, r), ratio))

    circle_detect_ratios = sorted(circle_detect_ratios, key=lambda p: -p[1])

    circles = []
    for (circle, ratio) in circle_detect_ratios:
        if all([distant(circle, other_circle) for other_circle in circles]):
            circles.append(circle)
            print(circle, ratio)

    result = Image.new("RGB", image.size)
    result.paste(image)
    draw_result = ImageDraw.Draw(result)

    for (a, b, r) in circles:
        draw_result.ellipse((a - r, b - r, a + r, b + r), outline=(255, 0, 0))

    return result


if __name__ == "__main__":

    image = Image.open("samples/sample1.png")
    result = pattern_detect(image)
    result.show()
