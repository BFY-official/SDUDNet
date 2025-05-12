import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from net import DNN
import torchvision
from torchvision import transforms
import cv2
from matplotlib import pyplot as plt




dnn = DNN.DNN()
dnn.load_state_dict(torch.load('models/synthetic.pth'))
dnn = dnn.cuda()

z = torch.randn(size=(1, 1, 256, 256))

sar = Image.open('my_datasets/S4_L1.png')

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), 
                    # torchvision.transforms.RandomCrop(128)
                    ])

transform1 = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), 
                    # torchvision.transforms.RandomCrop(128)
                    ])

sar = transform1(sar)

sar = sar.reshape(1, 1, 256, 256)

sar = sar.cuda()


desp, _ = dnn(sar)

desp_image = desp.squeeze().cpu().detach().numpy()

cv2.imwrite('results/result.png', desp_image*255)



plt.imshow(desp_image, cmap='gray')
plt.axis('off')
plt.show()


