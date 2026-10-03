"""Fotoğraf gerektirmeyen, bilinen perspektif dönüşümlü sentetik demo."""
from pathlib import Path
import cv2
import numpy as np
from main import write_image


def create_demo(destination=Path('examples')):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    world = np.full((480, 1100, 3), (34, 27, 21), np.uint8)
    for i in range(190):
        x, y = rng.integers(15, 1080), rng.integers(20, 455)
        color = tuple(int(v) for v in rng.integers(70, 240, 3))
        radius = int(rng.integers(3, 17))
        cv2.circle(world, (int(x), int(y)), radius, color, -1)
        if i % 3 == 0:
            cv2.putText(world, str(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.rectangle(world, (490, 305), (565, 360), (0, 0, 0), -1)
    cv2.putText(world, 'PANORAMA / SIFT + FLANN', (45, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1.15, (245, 235, 210), 2, cv2.LINE_AA)
    cv2.putText(world, 'Known perspective / reproducible demo', (400, 420),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (140, 215, 245), 2, cv2.LINE_AA)
    left = world[:, :720].copy()
    right_crop = world[:, 380:].copy()
    src = np.float32([[0, 0], [719, 0], [719, 479], [0, 479]])
    dst = np.float32([[14, 13], [698, 5], [711, 465], [4, 472]])
    P = cv2.getPerspectiveTransform(src, dst)
    right = cv2.warpPerspective(right_crop, P, (720, 480))
    # Sağ görüntü p -> inv(P) -> sağ kırpım -> +380 -> sol koordinatı.
    expected_H = np.float64([[1, 0, 380], [0, 1, 0], [0, 0, 1]]) @ np.linalg.inv(P)
    expected_H /= expected_H[2, 2]
    write_image(destination / 'left.png', left)
    write_image(destination / 'right.png', right)
    write_image(destination / 'reference.png', world)
    np.savetxt(destination / 'expected_homography.txt', expected_H)
    return left, right, world, expected_H


if __name__ == '__main__':
    create_demo()
    print('Sentetik örnekler examples/ klasörüne kaydedildi.')
