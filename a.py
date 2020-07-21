# import torch
# from torch.autograd import Variable

# x = Variable(torch.rand(2,3).float())
# print(x.data[0])

import os
import utils
import models
import torch.nn as nn
import torch
from torch.autograd import Variable
import torchvision.utils as vutils

criterion_itself = utils.Itself_loss()
output = Variable(torch.rand(2,3).float())
errG1 = criterion_itself(output)

print(errG1.item())