from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets,transforms
import torchvision
from torchsummary import summary
from tqdm import tqdm

from albumentations import  ( 
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose , Normalize ,ToFloat, Cutout
)

import cv2

import numpy as np

from albumentations.pytorch import  ToTensor 


def downloading_data_transforms_albumentations(data_set):
  class album_Compose_train:
      def __init__(self):
          self.albumentations_transform_train = Compose([
            HorizontalFlip(),
            
            Cutout(),
            # CLAHE(),
            Normalize(
              mean=[0.5, 0.5, 0.5],
              std=[0.5, 0.5, 0.5],
            ),
            ToTensor()
          ])

      def __call__(self,img):
        img = np.array(img)
        img = self.albumentations_transform_train(image=img)
        return img['image']

  class album_Compose_test:
      def __init__(self):
          self.albumentations_transform_test = Compose([
            Normalize(
              mean=[0.5, 0.5, 0.5],
              std=[0.5, 0.5, 0.5],
            ),
            ToTensor()
          ])

      def __call__(self,img):
        img = np.array(img)
        img = self.albumentations_transform_test(image=img)
        return img['image']

  trainset = data_set(root='./data', train=True, download=True,transform=album_Compose_train())
  testset = data_set(root='./data', train=False, download=True,transform=album_Compose_test())

  print('No.of images in train set are',len(trainset))
  print('No.of images in test set are',len(testset))

  return trainset,testset

