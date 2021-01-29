import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from bert_model.utils import constant

def sta(logits, label):
    logits = torch.argmax(logits, dim=1)
    # lo_tmp = torch.where(logits!=constant.LABEL_TO_ID['-2'], torch.ones_like(logits), torch.zeros_like(logits))
    # la_tmp = torch.where(label!=constant.LABEL_TO_ID['-2'], torch.ones_like(label), torch.zeros_like(label))
    right_num = (torch.where(logits==label, torch.ones_like(logits), torch.zeros_like(logits))*la_tmp).sum().item()
    logits_num = torch.where(logits!=constant.LABEL_TO_ID['-2'], torch.ones_like(logits), torch.zeros_like(logits)).sum().item()
    label_num = torch.where(label!=constant.LABEL_TO_ID['-2'], torch.ones_like(label), torch.zeros_like(label)).sum().item()
    return right_num, logits_num, label_num
    
