import cv2
import numpy as np
import time
#import os


def mask_it(img, lw, up):
    mask = cv2.inRange(img, lw, up)
    result = cv2.bitwise_and(img, img, mask=mask)
    return result


def region_of_interest(img):
    height = img.shape[0]
    width = img.shape[1]
    mask = np.ones_like(img)*255

    poly = np.array([[  # Ploygon to build custom mask
        (0, 0),
        (width, 0),
        (width, 210),
        (0, 210), ]], np.int32)
    masked = cv2.fillPoly(mask, poly, 0)  # return none --> fills region
    # Ou exclusivo para ignorar oq estiver fora da mask
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


i = 0
mean_time = 0
cap = cv2.VideoCapture("pista2.MP4")
while(cap.isOpened()):
    start = time.time()
    i = i+1
    _, frame = cap.read()
    dst = cv2.GaussianBlur(frame, (7, 7), 0)
    length, width, ch = frame.shape
    crop = frame[250:length, 0:200, :]

    # l1=(100,100,100)
    # up1=(101,120,120)
    l1 = (95, 95, 95)
    up1 = (101, 120, 120)  # Still in  RGB space!

    l2 = (0, 0, 60)
    up2 = (180, 40, 120)  # Still in  RGB space!

    trim = region_of_interest(frame)
    # hsv = cv2.cvtColor(trim, cv2.COLOR_BGR2HSV)
    # cv2.imshow('hsv',hsv)
    mask = mask_it(trim, l1, up1)
    # mask =mask_it(hsv,l2,up2)
    cv2.imshow('mask_it', mask)
    kernel = np.ones((7, 7), np.uint8)
    # opening = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel, iterations = 3)
    close = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=7)
    op = cv2.morphologyEx(close, cv2.MORPH_OPEN, kernel, iterations=8)
    # Se eu continuar, fecha a região certinho =)
    combo = cv2.addWeighted(frame, 0.8, op, 1, 1)


################ Calculate the  Moments and Visualize ################

    # convert image to grayscale image
    gray_mask = cv2.cvtColor(op, cv2.COLOR_BGR2GRAY)
    # convert the grayscale image to binary image
    ret, thresh = cv2.threshold(gray_mask, 0, 255, 0)

    # calculate moments of binary image
    M = cv2.moments(thresh)

    # calculate x,y coordinate of center
    try:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    except Exception:
        cX = 0
        cY = 0
    # put text and highlight the center

    int(frame.shape[1]/2), frame.shape[0]
    cX, cY
    # m = (y-y0)/(x-x0)  = tan(theta)
    rad = np.arctan((cY-frame.shape[0])/(cX-int(frame.shape[1]/2)+0.001))
    theta = np.degrees(rad)

    # print('Direção: %.5s graus' % theta)
    circle = np.zeros_like(frame)
    cv2.circle(circle, (cX, cY), 5, (255, 255, 255), -1)
    cv2.putText(circle, "centroid", (cX - 25, cY - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(circle, "Direction: %.5s graus" % theta, (25, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # write the lines
    cv2.line(circle, (cX, cY),
             (int(frame.shape[1]/2), frame.shape[0]), (255, 0, 0), 2)
    # display the image
    combo2 = cv2.addWeighted(combo, 0.9, circle, 1, 0)
    cv2.imshow('mask', op)
    cv2.imshow('resultado', combo2)


# Calculate the time taken to process the frame
    end = time.time()
    mean_time = (mean_time*(i-1) + (end - start))/i
    mean_fps = 1/mean_time

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("O tempo de execução médio foi de {:.2f} ms, ou {:.2f} FPS.".format(
            mean_time*1000, mean_fps))
        break
cap.release()
cv2.destroyAllWindows()
