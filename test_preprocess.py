import cv2
from omr_core.preprocess import preprocess_image

img = cv2.imread("sample.png")  # foto LJK lu
thresh = preprocess_image(img)

cv2.imshow("THRESH", thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()
