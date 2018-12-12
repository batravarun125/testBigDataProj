import numpy as np
import torch.nn.functional as F
import torch
def softmax(x):
    #print ((x.shape))
    a = F.softmax(x,dim=0)
    return a.max()
x = torch.tensor([1.2, 3])
a = softmax(x)
print ("k")
