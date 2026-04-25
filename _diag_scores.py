import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from omr_core.preprocess import preprocess_for_markers, preprocess_for_answers
from omr_core.detect_sheet import find_paper
import cv2, numpy as np

img = cv2.imread('sample 2.png')
thresh = preprocess_for_markers(img)
warped, M = find_paper(thresh)
src_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
warped_gray = cv2.warpPerspective(src_gray, M, (1000, 1414))
warped_ready = preprocess_for_answers(warped_gray)

y_start = int(1414 * 0.222)
y_end   = int(1414 * 0.975)
c2_xs = int(1000 * 0.590)
c2_xe = int(1000 * 0.855)

col_w = c2_xe - c2_xs
col_h = y_end - y_start
roi = warped_ready[y_start:y_end, c2_xs:c2_xe]
y_cuts = np.linspace(0, col_h, 15 + 1, dtype=int)

print('=== SCORES FOR Q26-Q30 ===')
for d in range(10, 15):
    yt = y_cuts[d]
    yb = y_cuts[d+1]
    scores = []
    for (xs_r, xe_r) in [(0.0,0.2),(0.2,0.4),(0.4,0.6),(0.6,0.8),(0.8,1.0)]:
        xt = int(col_w * xs_r)
        xb = int(col_w * xe_r)
        cell = roi[yt+4:yb-4, xt+4:xb-4]
        scores.append(255.0 - np.mean(cell))
    mean = np.mean(scores)
    std = np.std(scores)
    thresh = mean + 1.0 * std
    print(f'Q{15+d+1}: {["{:.1f}".format(s) for s in scores]} | Mean: {mean:.1f}, Std: {std:.1f}, Thresh: {thresh:.1f}')
