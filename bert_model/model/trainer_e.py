"""
A trainer class.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


from model.AspModel import AspModel
from utils import constant
from utils.scorer import sta
import utils.constant
LABEL_TO_ID = {'-1': 0, '0': 1, '1': 2, '-2': 3}
ID_TO_LABEL = {0:'-1', 1:'0', 2:'1', 3:'-2'}


ASP_TO_ID = utils.constant.ASP_TO_ID 
ID_TO_ASP = {v: i for i, v in ASP_TO_ID.items()}

class Trainer(object):
    def __init__(self, opt, emb_matrix=None):
        raise NotImplementedError

    def update(self, batch):
        raise NotImplementedError

    def predict(self, batch):
        raise NotImplementedError

    def update_lr(self, new_lr):
        torch_utils.change_lr(self.optimizer, new_lr)

    def load(self, filename):
        try:
            checkpoint = torch.load(filename,map_location=torch.device('cpu'))
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.model.load_state_dict(checkpoint['model'])
        self.opt = checkpoint['config']

    def save(self, filename):
        params = {
                'model': self.model.state_dict(),
                'config': self.opt,
                }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")

# 0: tokens, 1: mask_s, 2: label
def unpack_batch(batch, cuda):
    inputs, label = batch[0:2], batch[2]
    if cuda:
        inputs = [Variable(i.cuda()) for i in inputs]
        label = Variable(label.cuda())
    else:
        inputs = [Variable(i) for i in inputs]
        label = Variable(label)
    return inputs, label

def error_analysis_result(logits, label, batch_raw):
    logits = torch.argmax(logits, dim=2)
    for i in range(logits.size(0)):
        pred_aspects, pred_polarities = [], []
        for j in range(logits.size(1)):
            if logits[i][j] != 3:
                pred_aspects.append(ID_TO_ASP[j])
                pred_polarities.append(ID_TO_LABEL[logits[i][j].item()])
        batch_raw[i]['pred_aspect'] = pred_aspects
        batch_raw[i]['pred_polarity'] = pred_polarities

# 0: tokens, 2: mask_s, 3: label
class MyTrainer(Trainer):
    def __init__(self, opt):
        self.opt = opt
        self.model = AspModel(opt)
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        
        if opt['cuda']:
            self.model.cuda()
        self.optimizer = torch.optim.Adam(self.parameters, lr=self.opt['lr'])

    def update(self, batch):
        inputs, label = unpack_batch(batch, self.opt['cuda'])
        # step forward
        self.model.train()
        self.optimizer.zero_grad()
        logits = self.model(inputs)
        loss = F.cross_entropy(logits.view(-1, len(constant.LABEL_TO_ID)), label.view(-1), reduction='sum') / label.size(0)
        # backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt['max_grad_norm'])
        self.optimizer.step()
        # loss value
        loss_val = loss.item()
        return loss_val

    def predict(self, batch, batch_raw):
        inputs, label = unpack_batch(batch, self.opt['cuda'])
        # forward
        self.model.eval()
        logits = self.model(inputs)
        # loss 
        loss = F.cross_entropy(logits.view(-1, len(constant.LABEL_TO_ID)), label.view(-1), reduction='sum') / label.size(0)
        # predict result
        right_num, logits_num, label_num = sta(logits, label)
        # error analysis
        error_analysis_result(logits, label, batch_raw)
        loss_val = loss.item()
        return loss_val, right_num, logits_num, label_num
