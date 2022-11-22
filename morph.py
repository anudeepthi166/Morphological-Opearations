#importing modules
import numpy as np
from matplotlib import pyplot as plt
import cv2

#creating blank image
blank_img=np.zeros((500,700,3))
plt.imshow(blank_img)

#writing text on blank image
img=cv2.putText(
blank_img,
text='ABCDE',
org=(50,300),
fontFace=cv2.FONT_HERSHEY_SIMPLEX,
fontScale=5,
color=(255,255,255),
thickness=25,)

plt.imshow(img,cmap='gray')
cv2.imwrite("orig.jpg",img)

#EROSION
k=np.ones((5,5),dtype=np.uint8)
e_img=cv2.erode(img,k)
plt.imshow(e_img)

k=np.ones((5,5),dtype=np.uint8)
e_img=cv2.erode(img,k,iterations=4)
plt.imshow(e_img)
cv2.imwrite("e4.jpg",e_img)

#DIALATION
d_img=cv2.dilate(e_img,k,iterations=5)
plt.imshow(d_img)
cv2.imwrite("d5.jpg",d_img)

#MORPH_OPEN
oimg=cv2.imread('o_img.png')
o=cv2.morphologyEx(oimg,cv2.MORPH_OPEN,k)
plt.imshow(o)
cv2.imwrite("open.jpg",o)

#MORPH_CLOSE
c=cv2.morphologyEx(c,cv2.MORPH_CLOSE,k)
plt.imshow(c)
cv2.imwrite("c9.jpg",c)

#MORPH_GRADIENT
g=cv2.morphologyEx(img,cv2.MORPH_GRADIENT,k)
plt.imshow(g)
cv2.imwrite("g.jpg",g)
