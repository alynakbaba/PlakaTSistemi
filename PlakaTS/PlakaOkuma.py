import cv2
import pytesseract
import matplotlib.pyplot as plt
#pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Plaka resmin görüntülenmesi
img = cv2.imread("TespitEdilenPlaka/plaka.jpg")

plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.show()

#gray formata dönüştürülmesi
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow('gray', gray)

#bulanıklaştırma
blur = cv2.GaussianBlur(gray, (3,3), 0)

cv2.imshow('blur', blur)

#eşikleme
thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

cv2.imshow('thresh', thresh)

#Gürültüyü gidermek ve görüntüyü ters çevirmek için Morph açık

#dikdörtgen yapıyı tanımlama
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

#karakterleri içine alır ve açık boşlukları kaldırır
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
cv2.imshow('opening', opening)

#görüntüyü ters çevirir (beyazı-siyah,siyahı-beyaz)
invert = 255 - opening
cv2.imshow('invert', invert)

#Plakayı okuma
text = pytesseract.image_to_string(invert, lang='eng', config='--psm 6')
print(text)

cv2.waitKey(0)
cv2.destroyAllWindows()