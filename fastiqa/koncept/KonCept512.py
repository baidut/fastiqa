import torch
import torch.nn as nn
from ._inceptionresnetv2 import inceptionresnetv2

class model_qa(nn.Module):
    def __init__(self,num_classes,**kwargs):
        super(model_qa,self).__init__()
        # base_model = inceptionresnetv2(num_classes=1000, pretrained='imagenet')
        base_model = inceptionresnetv2(num_classes=1000)
        self.base= nn.Sequential(*list(base_model.children())[:-1])
        self.fc = nn.Sequential(
            nn.Linear(1536, 2048),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(2048),
            nn.Dropout(p=0.25),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Dropout(p=0.25),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self,x):
        x = self.base(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def koncept512(pretrained=True):
  KonCept512 = model_qa(num_classes=1)
  if pretrained:
    # KonCept512.load_state_dict(torch.load('./KonCept512.pth'))
    pass
  return KonCept512
