import albumentations as A
import cv2
import os

images = os.listdir('Images/CHASE/CHASE/train/image') # replace with the base dataset image folder location
masks = os.listdir('Images/CHASE/CHASE/train/label') # replace with the base dataset mask folder location

HEIGHT = 960
WIDTH = 999
transform = A.Compose([
    A.OneOf([
        A.RandomSizedCrop(min_max_height=(500, HEIGHT), height=HEIGHT, width=WIDTH, p=0.3),
        A.PadIfNeeded(min_height=HEIGHT, min_width=WIDTH, p=0.4)
            ], p=1),
    A.Rotate(limit=90, p=1, border_mode=cv2.BORDER_CONSTANT),
    A.HorizontalFlip(p=0.3),
    A.VerticalFlip(p=0.4),
    A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=0, p=0.4),
    A.OneOf([
        A.ElasticTransform(alpha=60, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.3, border_mode=cv2.BORDER_CONSTANT),
        A.GridDistortion(p=0.4, border_mode=cv2.BORDER_CONSTANT),
        A.OpticalDistortion(distort_limit=1, shift_limit=0.3, p=1, border_mode=cv2.BORDER_CONSTANT)
             ], p=0.3),
    A.CLAHE(p=0.4),
    A.RandomRotate90(p=0.4),
    A.Transpose(p=0.1),
    A.OneOf([A.Blur(blur_limit=3,p=0.4),
             A.OpticalDistortion(p=0.3, border_mode=cv2.BORDER_CONSTANT),
             A.GridDistortion(p=0.4, border_mode=cv2.BORDER_CONSTANT),
             ], p=1.0),
    A.RandomGamma(p=0.3),
])

counter = 1
for i in range(0, 1):
    image = cv2.imread(f'Images/CHASE/CHASE/train/image/{images[i]}') # replace with the base dataset file location
    mask = cv2.imread(f'Images/CHASE/CHASE/train/label/{masks[i]}') # replace with the base dataset file location
    for j in range(5):
        transformed = transform(image=image,mask=mask)
        transformed_image = transformed["image"]
        transformed_masks = transformed['mask']
        number = '{:0>4}'.format(counter)
        cv2.imwrite(f'CHASE/augmented_images/CHASE_{number}.jpg', transformed_image) # replace with desired file save location
        cv2.imwrite(f'CHASE/augmented_labels/CHASE_{number}_mask.jpg', transformed_masks) # replace with desired file save location
        counter += 1

