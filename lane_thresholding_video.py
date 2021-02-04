import cv2
import matplotlib.pyplot as plt
import numpy as np
print(cv2.__version__)


def region_of_interest(canny):
    height = canny.shape[0]
    width = canny.shape[1]
    mask = np.zeros_like(canny)

    poly = np.array([[  # Polígono para fazer a máscara (feito sob medida)
        (0, height),
        (30, 210),
        (200, 205),
        (width, 250),
        (width, height), ]], np.int32)

    cv2.fillPoly(mask, poly, 255)  # return none --> preenche a região
    # Ou exclusivo para ignorar oq estiver fora da mask
    masked_image = cv2.bitwise_and(canny, mask)
    return masked_image


# The problem is with my video?
s = 1
cap = cv2.VideoCapture("pista2.mp4")  # colocar o vídeo

while(cap.isOpened()):
    ret, pista = cap.read()  # Iniciando video
    print(f"frame {s} ")
    s = s+1
    gray = cv2.cvtColor(pista, cv2.COLOR_BGR2GRAY)  # transforma para cinza
    blur4 = cv2.GaussianBlur(gray, (9, 9), 0)  # aplicar blur
    # aplicar canny edges detector (um conjunto de operações)
    canny4 = cv2.Canny(blur4, 50, 150)

    # mascarar a região de interesse(na imagem foi feito no olho)
    a = region_of_interest(canny4)
    kernel = np.ones((3, 3), np.uint8)
    # Suavizar e juntar as linhas
    opening = cv2.morphologyEx(a, cv2.MORPH_CLOSE, kernel, iterations=5)
    linhas = cv2.HoughLinesP(opening, 2, np.pi/180, 100,
                             np.array([]), minLineLength=15, maxLineGap=10)
    # line_fit = []
    # if linhas is not None:
    #     for linha in linhas:
    #         for x1, y1, x2, y2 in linha:
    #             fit = np.polyfit((x1, x2), (y1, y2), 1)
    #             slope = fit[0]
    #             intercept = fit[1]
    #             line_fit.append((slope, intercept))

    # reta_media = np.average(line_fit, axis=0)

    # slope, intercept = reta_media
    # y1 = 350  # opening.shape[0]
    # y2 = int(y1*3/5)
    # x1 = int((y1-intercept)/slope)
    # x2 = int((y2-intercept)/slope)
    # coord = [x1, x2, y1, y2]
    line_image = np.zeros_like(pista)
    if linhas is not None:
        for linha in linhas:
            for x1, y1, x2, y2 in linha:
                line_image = cv2.line(
                    line_image, (x1, y1), (x2, y2), (255, 0, 0), 3)
                # Can iterate over the image to increase the number of lines!
    imagem = cv2.addWeighted(pista, 0.8, line_image, 1, 1)
    cv2.imshow('resultado', imagem)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
