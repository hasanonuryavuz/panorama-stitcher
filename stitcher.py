"""SIFT/FLANN ile sağ görüntüyü sol görüntünün koordinatlarına taşı."""
from dataclasses import dataclass
import cv2
import numpy as np


class StitchingError(ValueError):
    """Görüntülerden güvenilir bir panorama üretilemedi."""


@dataclass
class StitchResult:
    panorama: np.ndarray
    matches_image: np.ndarray
    homography: np.ndarray
    metrics: dict


def stitch(left, right, *, ratio=0.75, ransac=4.0, min_matches=12,
           min_inlier_ratio=0.35, max_canvas_pixels=12_000_000):
    """BGR uint8 görüntüler alır; homografi right -> left yönündedir.

    Eşleşmeler ve hata eşikleri işlenen görüntü çözünürlüğündedir.
    Siyah pikselleri boşluk saymak yerine ayrı geçerlilik maskeleri kullanır.
    """
    if not 0 < ratio < 1 or ransac <= 0 or min_matches < 4:
        raise StitchingError('Oran 0–1 arasında, RANSAC > 0, eşleşme sayısı >= 4 olmalı.')
    if not 0 < min_inlier_ratio <= 1 or max_canvas_pixels <= 0:
        raise StitchingError('Geçersiz güven veya tuval sınırı.')
    for img in (left, right):
        if (not isinstance(img, np.ndarray) or img.dtype != np.uint8
                or img.ndim != 3 or img.shape[2] != 3 or img.size == 0):
            raise StitchingError('Boş olmayan, 3 kanallı uint8 BGR görüntü gerekiyor.')

    # 1. SIFT: konumları (keypoints) ve çevrelerinin tanımlarını (descriptors) çıkar.
    sift = cv2.SIFT_create(nfeatures=5000)
    kp_l, desc_l = sift.detectAndCompute(cv2.cvtColor(left, cv2.COLOR_BGR2GRAY), None)
    kp_r, desc_r = sift.detectAndCompute(cv2.cvtColor(right, cv2.COLOR_BGR2GRAY), None)
    if desc_l is None or desc_r is None or min(len(desc_l), len(desc_r)) < 2:
        raise StitchingError('Yeterli özellik bulunamadı. Daha dokulu fotoğraflar kullanın.')

    # 2. FLANN: sağdaki her tanım için soldaki en yakın iki adayı bul.
    matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64))
    pairs = matcher.knnMatch(desc_r, desc_l, k=2)
    good = [pair[0] for pair in pairs
            if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance]
    # Tek sol noktaya yığılan birden çok eşleşme modelin güvenini şişirmesin.
    unique = {}
    for match in sorted(good, key=lambda m: m.distance):
        unique.setdefault(match.trainIdx, match)
    good = list(unique.values())
    if len(good) < min_matches:
        raise StitchingError(f'Yetersiz eşleşme: {len(good)} / {min_matches}. Ortak alanı artırın.')

    # 3. Homografi: kaynak sağ, hedef sol. RANSAC aykırı eşleşmeleri eler.
    src = np.float32([kp_r[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp_l[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, inliers = cv2.findHomography(src, dst, cv2.RANSAC, ransac)
    if H is None or inliers is None or not np.isfinite(H).all() or abs(H[2, 2]) < 1e-12:
        raise StitchingError('Homografi hesaplanamadı.')
    H = H / H[2, 2]
    inlier_count = int(inliers.sum())
    if inlier_count < 8 or inlier_count / len(good) < min_inlier_ratio:
        raise StitchingError('Eşleşmeler tutarlı değil; görüntüler aynı sahneyi göstermeyebilir.')
    h_l, w_l = left.shape[:2]
    h_r, w_r = right.shape[:2]
    corners_r = np.float32([[0, 0], [w_r, 0], [w_r, h_r], [0, h_r]])
    denominators = np.c_[corners_r, np.ones(4)] @ H[2, :]
    if np.any(np.abs(denominators) < 1e-8) or np.any(denominators > 0) != np.all(denominators > 0):
        raise StitchingError('Dönüşüm görüntü içinde sonsuza gidiyor; farklı fotoğraflar deneyin.')
    warped_corners = cv2.perspectiveTransform(corners_r.reshape(-1, 1, 2), H).reshape(-1, 2)
    corners_l = np.float32([[0, 0], [w_l, 0], [w_l, h_l], [0, h_l]])
    all_corners = np.vstack([corners_l, warped_corners])
    if not np.isfinite(all_corners).all():
        raise StitchingError('Geçersiz tuval koordinatları.')
    x0, y0 = np.floor(all_corners.min(axis=0)).astype(np.int64)
    x1, y1 = np.ceil(all_corners.max(axis=0)).astype(np.int64)
    width, height = int(x1 - x0), int(y1 - y0)
    if width <= 0 or height <= 0 or width * height > max_canvas_pixels:
        raise StitchingError('Tuval sınırı aşıldı; fotoğrafları küçültün veya eşleşmeleri kontrol edin.')
    # Negatif koordinatları tuval içinde tutmak için öteleme ekle.
    T = np.float64([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]])
    warped = cv2.warpPerspective(right, T @ H, (width, height))
    mask_r = cv2.warpPerspective(np.full((h_r, w_r), 255, np.uint8), T @ H,
                                 (width, height), flags=cv2.INTER_NEAREST)
    canvas_l = np.zeros_like(warped)
    mask_l = np.zeros((height, width), np.uint8)
    canvas_l[-y0:-y0 + h_l, -x0:-x0 + w_l] = left
    mask_l[-y0:-y0 + h_l, -x0:-x0 + w_l] = 255
    if not np.any((mask_l > 0) & (mask_r > 0)):
        raise StitchingError('Dönüşüm sonrası görüntüler örtüşmüyor.')

    # 4. Feather blending: sınıra uzak pikseller daha yüksek ağırlık alır.
    def weights(mask):
        padded = np.pad(mask, 1)
        distance = cv2.distanceTransform(padded, cv2.DIST_L2, 3)[1:-1, 1:-1]
        return distance + (mask > 0).astype(np.float32) * 1e-3

    wl, wr = weights(mask_l), weights(mask_r)
    total = np.maximum(wl + wr, 1e-6)
    panorama = np.clip((canvas_l.astype(np.float32) * wl[..., None]
                       + warped.astype(np.float32) * wr[..., None]) / total[..., None],
                      0, 255).round().astype(np.uint8)
    union = cv2.bitwise_or(mask_l, mask_r)
    x, y, w, h = cv2.boundingRect(union)
    panorama = panorama[y:y+h, x:x+w]
    match_view = cv2.drawMatches(right, kp_r, left, kp_l, good, None,
                                 matchesMask=inliers.ravel().tolist(),
                                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    projected = cv2.perspectiveTransform(src, H)
    error = np.linalg.norm(projected - dst, axis=2).ravel()[inliers.ravel().astype(bool)]
    return StitchResult(panorama, match_view, H, {
        'keypoints_left': len(kp_l), 'keypoints_right': len(kp_r),
        'good_matches': len(good), 'inliers': inlier_count,
        'inlier_ratio': round(inlier_count / len(good), 4),
        'median_reprojection_error_px': round(float(np.median(error)), 4),
        'output_width': panorama.shape[1], 'output_height': panorama.shape[0],
    })
