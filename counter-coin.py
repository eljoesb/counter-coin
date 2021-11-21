import cv2 as cv
import numpy as np

valor_gauss = 3
valor_kernel = 9

original = cv.imread('monedas.jpeg')
gris = cv.cvtColor(original, cv.COLOR_BGR2GRAY)
gauss = cv.GaussianBlur(gris, (valor_gauss, valor_gauss), 0)
canny = cv.Canny(gauss, 0, 100)

kernel = np.ones((valor_kernel, valor_kernel), np.uint8)
cierre = cv.morphologyEx(canny, cv.MORPH_CLOSE, kernel)

contorno, jerarquia = cv.findContours(cierre.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

print(f'Monedas encontradas {len(contorno)}')

cv.drawContours(original, contorno, -1, (255,0,0), 2)

cv.imshow('Original', original)
# cv.imshow('gris', gris)
# cv.imshow('gauss', gauss)
# cv.imshow('canny', canny)

cv.waitKey(0)