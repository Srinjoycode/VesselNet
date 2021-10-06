import os
import cv2
from matplotlib import pyplot as plt
  
# create figure
fig = plt.figure(figsize=(10, 7))
  
# setting values to rows and column variables
rows = 4
columns = 3

#listing image files
files = os.listdir()

no = 4

#Displaying the original image
Image1 = cv2.imread()
Image1 = cv2.cvtColor(Image1, cv2.COLOR_BGR2RGB)
fig.add_subplot(rows, columns, 2)
plt.imshow(Image0)
plt.axis('off')
plt.title("ORIGINAL", fontsize=20)


for i in range(1, len(files)+1):
    Image = cv2.imread()
    Image = cv2.cvtColor(Image, cv2.COLOR_BGR2RGB)
    fig.add_subplot(rows, columns, no)
    no += 1
    plt.imshow(Image1)
    plt.axis('off')
    plt.title("Augmentaion name", fontsize=10)
