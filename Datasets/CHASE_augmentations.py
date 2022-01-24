import albumentations as A
import cv2
import os

images = os.listdir('Images/CHASE/CHASE/train/image') # replace with the base dataset image folder location
masks = os.listdir('Images/CHASE/CHASE/train/label') # replace with the base dataset mask folder location

HEIGHT=960
WIDTH=999
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
for i in range(0, 1):
    image = cv2.imread(f'Images/CHASE/CHASE/train/image/{images[i]}') # replace with the base dataset file location
    mask = cv2.imread(f'Images/CHASE/CHASE/train/label/{masks[i]}') # replace with the base dataset file location
    for j in range(15):
        transformed = transform(image=image,mask=mask)
        transformed_image = transformed["image"]
        transformed_masks = transformed['mask']
        number = '{:0>4}'.format(counter)
        cv2.imwrite(f'CHASE/augmented_images/CHASE_{number}.jpg', transformed_image) # replace with desired file save location
        cv2.imwrite(f'CHASE/augmented_labels/CHASE_{number}_mask.jpg', transformed_masks) # replace with desired file save location
        counter += 1

