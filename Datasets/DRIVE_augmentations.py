import albumentations as A
import cv2
import os
from PIL import Image
import numpy as np

images = os.listdir('Images/DRIVE/DRIVE/train/images') # replace with the base dataset image folder location
masks = os.listdir('Images/DRIVE/DRIVE/train/labels') # replace with the base dataset mask folder location

HEIGHT = 584
WIDTH = 565
transform = A.Compose([
    A.OneOf([
        A.RandomSizedCrop(min_max_height=(100, HEIGHT), height=HEIGHT, width=WIDTH, p=1),
                         A.PadIfNeeded(min_height=HEIGHT, min_width=WIDTH, p=1)
            ], p=1),
    A.Rotate(limit=90, p=1, border_mode=cv2.BORDER_CONSTANT),
    A.HorizontalFlip(p=1),
    A.VerticalFlip(p=1),
    A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=0, p=1),
    A.CLAHE(p=1),
    A.RandomRotate90(p=1),
    A.Blur(blur_limit=3,p=1),
    A.RandomGamma(p=1),
    A.RandomScale(scale_limit=(0, 1), p=1, interpolation=1), A.PadIfNeeded(HEIGHT, WIDTH, border_mode=cv2.BORDER_CONSTANT),
    A.Resize(height = HEIGHT, width = WIDTH, interpolation=1, always_apply=False, p=1)])



counter = 1
for i in range(0, len(images)):
    image = cv2.imread(f'Images/DRIVE/DRIVE/train/images/{images[i]}') # replace with the base dataset file location
    mask = Image.open(f'Images/DRIVE/DRIVE/train/labels/{masks[i]}') # replace with the base dataset file location
    mask = np.array(mask)
    for j in range(15):
        transformed = transform(image=image, mask=mask)
        transformed_image = transformed["image"]
        transformed_masks = transformed['mask']
        number = '{:0>4}'.format(counter)
        cv2.imwrite(f'DRIVE/augmented_images/DRIVE_{number}.jpg', transformed_image) # replace with desired file save location
        final_mask = Image.formarray(transformed_masks)
        final_mask.save(f'CHASE/augmented_labels/DRIVE_{number}_mask.jpg') # replace with desired file save location
        counter += 1
