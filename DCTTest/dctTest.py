import numpy as np
import cv2
save_dir = 'D:\\pythonStudy\\secretPic\\'
y = cv2.imread('D:\\pythonStudy\\orl_face\\s1\\1.pgm',0)
y1 = y.astype(np.float32)
Y = cv2.dct(y1)
cv2.imshow("Dct",Y)
y2 = cv2.idct(Y)
cv2.imshow("iDCT",y2.astype(np.uint8))
cv2.waitKey(0)
cv2.imwrite(save_dir+'1.png', Y, 0)
cv2.destroyAllWindows()