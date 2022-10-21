import cv2
import os


path = r"projects/apple_disease_classification/data/Train/Normal_Apple"


def path_(path):

    images = []

    for filename in os.listdir(path):

        img = cv2.imread(os.path.join(path, filename))

        if img is not None:

            images.append(img)

    print(len(images))

    return images


path_(path)
