import numpy as np
import cv2
import matplotlib.pyplot as plt


def region_of_interest(img):
    height = img.shape[0]
    width = img.shape[1]
    mask = np.ones_like(img)*255

    poly = np.array([[  # Polígono para fazer a máscara (feito sob medida da)
        (0, 0),
        (width, 0),
        (width, 210),
        (0, 210), ]], np.int32)
    masked = cv2.fillPoly(mask, poly, 0)  # return none --> preenche a região
    # Ou exclusivo para ignorar oq estiver fora da mask
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


cap = cv2.VideoCapture(
    "C://Users//Paulo Rodrigues//Desktop//Self-Driving Cars Course//test2.mp4")
# print("passou")
while(cap.isOpened()):
    # frame by frame of video
    ret, pista = cap.read()  # creating empty image of same size
    image = cv2.GaussianBlur(pista, (15, 15), 0)
    height, width, no_use = image.shape
    # Fazer uma cópia das dimensões da imagem
    empty_img = np.zeros((height, width), np.uint8)

# APPLY K-MEANS CLUSTERING:

    Z = image.reshape((-1, 3))  # flatten the image
    # need to convert to np.float32
    Z = np.float32(Z)
    # define criteria,
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                8, 1.0)  # type, max iteration, epsilon
    K = 4  # number of clusters required(K)   -- Episilon = required accuracy
    flags = cv2.KMEANS_RANDOM_CENTERS
    ret, label, center = cv2.kmeans(
        Z, K, None, criteria, 8, flags)  # apply kmeans()
    # Args:  Attempts - Number of initializations to get the best compactiness
    # Flags: can be  cv2.KMEANS_PP_CENTERS or cv2.KMEANS_RANDOM_CENTERS

    # OUTPUTS: Compactiness d^2 of the centers , label array, centers

    # Now convert back into uint8, and make original image  # Reconstruct image
    center = np.uint8(center)
    res = center[label.flatten()]  # Reconstruct the image data
    res2 = res.reshape((image.shape))

    # CONVERTED TO A LUV IMAGE AND MADE EMPTY IMAGE, A MASK
    blur = cv2.GaussianBlur(res2, (15, 15), 0)
    kernel = np.ones((3, 3), np.uint8)
    blur = cv2.morphologyEx(blur, cv2.MORPH_CLOSE, kernel, iterations=3)

    gray = cv2.cvtColor(blur, cv2.COLOR_RGB2GRAY)
    LUV = cv2.cvtColor(blur, cv2.COLOR_RGB2LUV)
    l = LUV[:, :, 0]
    v1 = l > 80
    v2 = l < 150
    value_final = v1 & v2
    empty_img[value_final] = 255
    empty_img[LUV[:, :100, :]] = 0
    final_masked = cv2.line(empty_img, (40, height), (400, height), 255, 120)
    final_mask = region_of_interest(final_masked)

    cv2.namedWindow('original', cv2.WINDOW_NORMAL)
    cv2.imshow('original', pista)
    cv2.namedWindow('tried_extraction', cv2.WINDOW_NORMAL)
    cv2.imshow('tried_extraction', final_mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
