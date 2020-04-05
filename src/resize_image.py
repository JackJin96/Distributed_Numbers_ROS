import cv2
import os

for filename in os.listdir("./test_images"):
    src = "./test_images/" + filename
    img = cv2.imread(src)
    imgcp = img.copy()
    print(imgcp.shape)
    imgcp = cv2.resize(imgcp, None, fx=0.25, fy=0.25)
    print(imgcp.shape)
    cv2.imwrite(src, imgcp)