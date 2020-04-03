
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
from torchsummary import summary


from albumentations import ( 
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose
)

def downloading_data(data_set):

  
  train_transform = transforms.Compose(
    [transforms.ToTensor(),
     HorizontalFlip(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

  test_transform = transforms.Compose(
    [transforms.ToTensor(),
    #  transforms.RandomRotation((-11.0, 11.0)),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

  trainset = data_set(root='./data', train=True,
                                        download=True, transform=train_transform)
  testset = data_set(root='./data', train=False,
                                       download=True, transform=test_transform)

  print('No.of images in train set are',len(trainset))
  print('No.of images in test set are',len(testset))
  return trainset,testset


def loading_to_train_test_loader(SEED,traindata,testdata):
  cuda = torch.cuda.is_available()
  print("CUDA Available?", cuda)
  # For reproducibility
  torch.manual_seed(SEED)

  if cuda:
      torch.cuda.manual_seed(SEED)

      trainloader = torch.utils.data.DataLoader(traindata, batch_size=128,
                                            shuffle=True, num_workers=4)

      testloader = torch.utils.data.DataLoader(testdata, batch_size=128,
                                          shuffle=False, num_workers=4)
      
   
  classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
  print('Train and Test data loaded.......')
  return trainloader,testloader


def Build_your_ResNetmodel(params):
    

  class BasicBlock(nn.Module):
      expansion = 1

      def __init__(self, in_planes, planes, stride=1):
          super(BasicBlock, self).__init__()
          self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
          self.bn1 = nn.BatchNorm2d(planes)
          self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
          self.bn2 = nn.BatchNorm2d(planes)

          self.shortcut = nn.Sequential()
          if stride != 1 or in_planes != self.expansion*planes:
              self.shortcut = nn.Sequential(
                  nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                  nn.BatchNorm2d(self.expansion*planes)
              )

      def forward(self, x):
          out = F.relu(self.bn1(self.conv1(x)))
          out = self.bn2(self.conv2(out))
          out += self.shortcut(x)
          out = F.relu(out)
          return out

  class ResNet(nn.Module):
      def __init__(self, block, num_blocks, num_classes=10):
          super(ResNet, self).__init__()
          self.in_planes = 64

          self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
          self.bn1 = nn.BatchNorm2d(64)
          self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
          self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
          self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
          self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
          self.linear = nn.Linear(512*block.expansion, num_classes)

      def _make_layer(self, block, planes, num_blocks, stride):
          strides = [stride] + [1]*(num_blocks-1)
          layers = []
          for stride in strides:
              layers.append(block(self.in_planes, planes, stride))
              self.in_planes = planes * block.expansion
          return nn.Sequential(*layers)

      def forward(self, x):
          out = F.relu(self.bn1(self.conv1(x)))
          out = self.layer1(out)
          out = self.layer2(out)
          out = self.layer3(out)
          out = self.layer4(out)
          out = F.avg_pool2d(out, 4)
          out = out.view(out.size(0), -1)
          out = self.linear(out)
          return out

  return ResNet(BasicBlock, params)



def training_model(model, device, train_loader, optimizer, epochs):

  import torch
  import torch.nn as nn
  import torch.nn.functional as F
  import torch.optim as optim
  from torchvision import datasets, transforms
  import torchvision
  from torchsummary import summary
  from tqdm import tqdm
  train_losses = []
  test_losses = []
  train_acc = []
  test_acc = []
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
  
  for epoch in range(epochs):
    model.train()
    
    correct = 0
    processed = 0
    train_loss=0

    
    for data, target in train_loader:
      data, target = data.to(device), target.to(device)

      # Init
      optimizer.zero_grad()
      # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes. 
      # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

      # Predict
      y_pred = model(data)

      # Calculate loss
      loss = criterion(y_pred, target)
      train_losses.append(loss)

      # Backpropagation
      loss.backward()
      optimizer.step()

      # Update pbar-tqdm
      
      pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
      correct += pred.eq(target.view_as(pred)).sum().item()
      processed += len(data)
     
      train_acc.append(100*correct/processed)
    print('Epoch:',epoch)
    print('Train Accuracy=',100*correct/processed)
      
  return train_losses,train_acc



def testing_model(model, device, testloader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(testloader.dataset)
    test_acc=(100. * correct / len(testloader.dataset))


    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(testloader.dataset),test_acc))
    return test_loss, test_acc
    
    



