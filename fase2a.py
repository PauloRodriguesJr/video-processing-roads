import numpy as np
import cv2


def region_of_interest(img):
    height = img.shape[0]
    width = img.shape[1]
    mask = np.ones_like(img)*255

    poly = np.array([[  # Polígono para fazer a máscara (Precisa ser tunado)
        (0, 0),
        (width, 0),
        (width, 210),
        (0, 210), ]], np.int32)
    masked = cv2.fillPoly(mask, poly, 0)  # return none --> preenche a região
    # Ou exclusivo para ignorar oq estiver fora da mask
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


#cap = cv2.VideoCapture("pista1.MP4")
cap = cv2.VideoCapture(
    "C://Users//Paulo Rodrigues//Desktop//Self-Driving Cars Course//test2.mp4")  # colocar o vídeo

while(cap.isOpened()):
    _, pista = cap.read()

# Início

    gray = cv2.cvtColor(pista, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    #gray = gray[10:210,10:210]
    T, c = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU)
    sf = region_of_interest(c)

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(sf, cv2.MORPH_CLOSE, kernel, iterations=5)

    # liNES
    linhas = cv2.HoughLinesP(opening, 2, np.pi/180, 100,
                             np.array([]), minLineLength=10, maxLineGap=10)

    line_fit = []
#		return None
    for linha in linhas:
        if linhas is not None:
            for x1, y1, x2, y2 in linha:
                try:
                    fit = np.polyfit((x1, x2), (y1, y2), 1)
                    slope = fit[0]
                    intercept = fit[1]
                    line_fit.append((slope, intercept))
                except:
                    pass
    reta_media = np.average(line_fit, axis=0)

    slope, intercept = reta_media
    y1 = 350  # opening.shape[0]
    y2 = int(y1*3/5)
    x1 = int((y1-intercept)/slope)
    x2 = int((y2-intercept)/slope)
    coord = [x1, x2, y1, y2]

    line_image = np.zeros_like(pista)
    if linhas is not None:
        line_image = cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    imagem = cv2.addWeighted(pista, 0.8, line_image, 1, 1)

    cv2.imshow('result', imagem)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
