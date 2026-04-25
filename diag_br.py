import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import cv2, numpy as np
from omr_core.preprocess import preprocess_image

img = cv2.imread('sample 2.png')
PAD = 50
padded = cv2.copyMakeBorder(img, PAD, PAD, PAD, PAD, cv2.BORDER_CONSTANT, value=[255,255,255])
thresh = preprocess_image(padded)

ph, pw = thresh.shape[:2]
mid_x, mid_y = pw//2, ph//2

contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print('All BR zone contours (cx>=mid_x, cy>=mid_y):')
for c in contours:
    area = cv2.contourArea(c)
    if area < 100: continue
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04*peri, True)
    x, y, w, h = cv2.boundingRect(approx)
    if h == 0: continue
    ar = w / float(h)
    M = cv2.moments(c)
    if M['m00'] == 0: continue
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    if cx >= mid_x and cy >= mid_y:
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        sol = area / hull_area if hull_area > 0 else 0
        near_right = (x + w) > (pw * 0.78)
        near_bot = (y + h) > (ph * 0.78)
        marker = " <-- NEAR CORNER" if (near_right and near_bot) else ""
        print(f"  area={area:>8.0f}  ar={ar:.2f}  sol={sol:.2f}  cx={cx:>5}  cy={cy:>5}  verts={len(approx)}{marker}")
