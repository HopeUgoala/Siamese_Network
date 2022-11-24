# Importing the neccessary library
import os
import cv2
import time
start = time.time()
root_dir = 'C:/Users/IBE/Downloads/New folder/Dataset/Joel'
IMAGES = []
i=0
Cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
# Iterating through directory and subfolders
for subdir, dirs,files in os.walk(root_dir):
    for file in files:
        Image = cv2.imread(os.path.join(subdir, file))
        # Grayscale conversion
        gray = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
        # Feeding the grayscale to a classifier
        face = Cascade.detectMultiScale(
                gray,
                scaleFactor=1.3,
                minNeighbors=5 )
        for (x,y,w,h) in face:
            # Slicing out the face as mentioned above
            Image_crop = Image[y:y+h,x:x+h]
            #resizing the image to 150x150
            Image_crop = cv2.resize(Image_crop, (150, 150))
            # replacing the previous images with the extracted faces
            cv2.imwrite(os.path.join(subdir, file),Image_crop)
            # Append every extracted faces
            IMAGES.append(Image_crop)
end = time.time()
print(len(IMAGES), type(IMAGES), ((start-end)/60))