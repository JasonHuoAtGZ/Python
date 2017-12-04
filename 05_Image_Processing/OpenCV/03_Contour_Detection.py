import cv2
import numpy as np

'''
img = cv2.imread("D:/03_Document/04_OpenCV/HOG/Kyrie_Irving.jpg")
img = img[:,:,0]

ret, thresh = cv2.threshold(img, 127, 255, 0)
image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
img = cv2.drawContours(color, contours, -1, (0,255,0), 1)
cv2.imshow("contours", color)
cv2.waitKey()
cv2.destroyAllWindows()
'''


import cv2
import numpy as np
img = cv2.pyrUp(cv2.imread("D:/03_Document/04_OpenCV/HOG/102551.18183164_1000X1000.jpg", cv2.THRESH_BINARY_INV))
#img = cv2.imread("D:/03_Document/04_OpenCV/HOG/095309.18997357_1000X1000.jpg")

ret, thresh = cv2.threshold(cv2.cvtColor(img.copy(),cv2.COLOR_BGR2GRAY) , 127, 255, cv2.THRESH_BINARY)

image, contours, hier = cv2.findContours(thresh,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for c in contours:
    # find bounding box coordinates
    x,y,w,h = cv2.boundingRect(c)
    cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 1)

    '''
    # find minimum area
    rect = cv2.minAreaRect(c)
    # calculate coordinates of the minimum area rectangle
    box = cv2.boxPoints(rect)
    # normalize coordinates to integers
    box = np.int0(box)
    # draw contours
    cv2.drawContours(img, [box], 0, (0,0, 255), 3)


    # calculate center and radius of minimum enclosing circle
    (x,y),radius = cv2.minEnclosingCircle(c)
    # cast to integers
    center = (int(x),int(y))
    radius = int(radius)
    # draw the circle
    img = cv2.circle(img,center,radius,(0,255,0),2)
    '''

cv2.drawContours(img, contours, -1, (255, 0, 0), 1)
cv2.imshow("contours", img)
cv2.waitKey()
cv2.destroyAllWindows()