import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import cv2, numpy as np
from omr_core.preprocess import preprocess_for_markers

img = cv2.imread('sample 2.png')
thresh = preprocess_for_markers(img)
padding = 20
padded = cv2.copyMakeBorder(thresh, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=0)
h_img, w_img = padded.shape[:2]
mid_x, mid_y = w_img // 2, h_img // 2

contours, _ = cv2.findContours(padded, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
img_area = h_img * w_img

zones = {'TL': [], 'TR': [], 'BL': [], 'BR': []}
for c in contours:
    area = cv2.contourArea(c)
    if area < 50 or area > (img_area * 0.035):
        continue
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.05 * peri, True)
    if not (3 <= len(approx) <= 8):
        continue
    x, y, w, h = cv2.boundingRect(approx)
    if h == 0 or w == 0:
        continue
    ar = w / float(h)
    if not (0.4 <= ar <= 2.5):
        continue
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    if hull_area == 0:
        continue
    sol = area / hull_area
    if sol < 0.7:
        continue
    edge_w = w_img * 0.28
    edge_h = h_img * 0.28
    near_l = x < edge_w
    near_r = (x + w) > (w_img - edge_w)
    near_t = y < edge_h
    near_b = (y + h) > (h_img - edge_h)
    if not ((near_l or near_r) and (near_t or near_b)):
        continue
    M = cv2.moments(c)
    if M['m00'] == 0:
        continue
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    if cx < mid_x and cy < mid_y:
        zone = 'TL'
    elif cx >= mid_x and cy < mid_y:
        zone = 'TR'
    elif cx < mid_x and cy >= mid_y:
        zone = 'BL'
    else:
        zone = 'BR'
    zones[zone].append({'area': area, 'ar': ar, 'sol': sol, 'pos': (x, y), 'approx': approx})

print('=== ALL CANDIDATES PER ZONE (sample 2.png) ===')
for z in ['TL', 'TR', 'BL', 'BR']:
    candidates = sorted(zones[z], key=lambda c: c['area'], reverse=True)
    print(f'\n{z} ({len(candidates)} candidates):')
    for i, c in enumerate(candidates[:8]):
        tag = ' <-- LARGEST' if i == 0 else ''
        print(f"  area={c['area']:>7.0f}  ar={c['ar']:.2f}  sol={c['sol']:.2f}  pos={c['pos']}{tag}")
