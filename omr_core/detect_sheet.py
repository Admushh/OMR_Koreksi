import cv2
import numpy as np

def find_paper(thresh):
    """
    Detect paper using 4 black square markers at corners
    and apply perspective transform.
    """

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    markers = []

    for c in contours:
        area = cv2.contourArea(c)
        if area < 500:
            continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # marker = kotak (4 sisi)
        if len(approx) == 4 and area > 1000:
            markers.append(approx)

    if len(markers) < 4:
        # fallback: return original
        return thresh

    # ambil 4 marker terbesar
    markers = sorted(markers, key=cv2.contourArea, reverse=True)[:4]

    # ambil titik pusat masing-masing marker
    centers = []
    for m in markers:
        M = cv2.moments(m)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        centers.append((cx, cy))

    if len(centers) != 4:
        return thresh

    # urutkan marker: TL, TR, BR, BL
    rect = order_points(np.array(centers, dtype="float32"))

    # ukuran output
    width = int(max(
        np.linalg.norm(rect[0] - rect[1]),
        np.linalg.norm(rect[2] - rect[3])
    ))
    height = int(max(
        np.linalg.norm(rect[0] - rect[3]),
        np.linalg.norm(rect[1] - rect[2])
    ))

    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(thresh, M, (width, height))

    return warped


def order_points(pts):
    """
    Order points: top-left, top-right, bottom-right, bottom-left
    """
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # TL
    rect[2] = pts[np.argmax(s)]  # BR

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # TR
    rect[3] = pts[np.argmax(diff)]  # BL

    return rect
