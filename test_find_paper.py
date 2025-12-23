import cv2
import os
from omr_core.preprocess import preprocess_image
from omr_core.detect_sheet import find_paper

BASE_DIR = os.path.dirname(__file__)
IMG_PATH = os.path.join(BASE_DIR, "sample.png")

img = cv2.imread(IMG_PATH)
thresh = preprocess_image(img)
warped = find_paper(thresh)

cv2.imshow("THRESH", thresh)
cv2.imshow("WARPED", warped)
cv2.waitKey(0)
cv2.destroyAllWindows()
