import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytesseract

#resimlerin görüntülenmesi
resim_adresi = os.listdir("image")

img = cv2.imread("image/yCar1.jpg")
# img = cv2.resize(img,(500,500))   #bazı resimlerde boyutlar farklı olduğu için plaka okuması yapmadığı zaman yorum satırından çıkarılabilir.

plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.show()

#resimlerin gray formata dönüştürülmesi
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

plt.imshow(img_gray,cmap="gray")
plt.show()

#bulanıklaştırma
b_img = cv2.medianBlur(img_gray,5)
b_img = cv2.medianBlur(b_img,5)

plt.imshow(b_img,cmap="gray")
plt.show()

#kenar tespiti
edged = cv2.Canny(img_gray, 30, 150)
contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)

cv2.imshow('Edged', edged)


# Plakayı bulma
roi = None
for contour in contours:
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.015 * perimeter, True)
    if len(approx) == 4:
        roi = approx
        break

roi = np.array([roi], np.int32)
points = roi.reshape(4, 2)
x, y = np.split(points, [-1], axis=1)

(x1, x2) = (np.min(x), np.max(x))
(y1, y2) = (np.min(y), np.max(y))
plaka = img[y1:y2, x1:x2]
cv2.imshow('Plaka Tespiti', plaka)

#plaka okuması
#pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
text = pytesseract.image_to_string(plaka, lang='eng', config='--psm 6')
print(text)

#plakayı "TespitEdilenPlaka" dosyasına kaydetme
path_to_save = "TespitEdilenPlaka/"
file_name = "plaka.jpg"
file_path = path_to_save + file_name
cv2.imwrite(file_path, plaka)

cv2.waitKey(0)
cv2.destroyAllWindows()