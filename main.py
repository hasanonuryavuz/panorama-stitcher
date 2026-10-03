"""Komut satırı: python main.py --left ... --right ..."""
import argparse
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np
from stitcher import stitch, StitchingError


def read_image(path, max_side):
    # imdecode/tofile Türkçe karakter içeren Windows yollarını da destekler.
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise StitchingError(f'Görüntü okunamadı: {path}')
    scale = min(1.0, max_side / max(image.shape[:2]))
    if scale < 1:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return image


def write_image(path, image):
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise StitchingError(f'Görüntü yazılamadı: {path}')
    encoded.tofile(path)


def main():
    parser = argparse.ArgumentParser(description='SIFT + FLANN ile iki görüntüden panorama üretir.')
    parser.add_argument('--left', type=Path, required=True)
    parser.add_argument('--right', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('outputs'))
    parser.add_argument('--ratio', type=float, default=0.75)
    parser.add_argument('--ransac', type=float, default=4.0)
    parser.add_argument('--max-side', type=int, default=1400)
    args = parser.parse_args()
    try:
        if args.max_side <= 0:
            raise StitchingError('--max-side pozitif olmalı.')
        cv2.setRNGSeed(42)
        started = time.perf_counter()
        left, right = read_image(args.left, args.max_side), read_image(args.right, args.max_side)
        result = stitch(left, right, ratio=args.ratio, ransac=args.ransac)
        args.output.mkdir(parents=True, exist_ok=True)
        write_image(args.output / 'panorama.jpg', result.panorama)
        write_image(args.output / 'matches.jpg', result.matches_image)
        result.metrics['elapsed_seconds'] = round(time.perf_counter() - started, 3)
        result.metrics['homography_right_to_left'] = result.homography.tolist()
        (args.output / 'metrics.json').write_text(json.dumps(result.metrics, indent=2), encoding='utf-8')
        print(json.dumps(result.metrics, indent=2))
        print(f'Kaydedildi: {args.output.resolve()}')
        return 0
    except (StitchingError, OSError, cv2.error) as exc:
        print(f'Hata: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
