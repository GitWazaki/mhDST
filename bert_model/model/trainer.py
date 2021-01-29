"""
A trainer class.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from transformers import BertTokenizer

from bert_model.model.AspModel import AspModel
from bert_model.utils import constant
from bert_model.utils.scorer import sta


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
        # try:
        checkpoint = torch.load(filename,map_location=torch.device('cpu'))
        # except BaseException:
        #     print("Cannot load model from {}".format(filename))
        #     exit()
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

# 0: tokens, 2: mask_s, 3: label
class MyTrainer(Trainer):
    def __init__(self, opt):
        self.opt = opt
        self.model = AspModel(opt).cuda()
        # self.model = AspModel(opt)
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(self.parameters, lr=self.opt['lr'])
        self.tokenizer = BertTokenizer.from_pretrained(opt['bert_path'])
        self.tokenizer.add_special_tokens({'additional_special_tokens':opt['special_tokens']})
        self.threshold = 4
        self.num_class = opt['num_class']
    def update(self, batch):
        inputs, label = unpack_batch(batch, self.opt['cuda'])
        # step forward
        self.model.train()
        self.optimizer.zero_grad()
        logits = self.model(inputs)
        loss = F.cross_entropy(logits.view(-1, self.num_class), label.view(-1), reduction='sum') / label.size(0)
        
        # backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters, self.opt['max_grad_norm'])
        self.optimizer.step()
        
        # loss value
        loss_val = loss.item()
        
        return loss_val

    def predict(self, batch):
        inputs, label = unpack_batch(batch, self.opt['cuda'])
        # forward
        self.model.eval()
        logits = self.model(inputs)
        # print(logits.size())
        # print(logits)
        
        # loss 
        loss = F.cross_entropy(logits.view(-1, self.num_class), label.view(-1), reduction='sum') / label.size(0)
        # preds = torch.argmax(logits, dim=1).detach().cpu().numpy().tolist()
        scores,preds = torch.max(logits,dim=1)
        scores = scores.detach().cpu().numpy().tolist()
        preds = preds.detach().cpu().numpy().tolist()
        # preds = [ele if score > self.threshold else self.num_class for ele,score in zip(preds,scores)]
        label = label.detach().cpu().numpy().tolist()
        # predict result
        # right_num, logits_num, label_num = sta(logits, label)
        loss_val = loss.item()
        
        return loss_val, preds, scores,label

    def run_pred_single(self,sentence):
        tokens = self.tokenizer.tokenize(sentence)
        tokens = ['[CLS]']+tokens+['[SEP]']
        tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        # delete the 0 length sentences 
        l = len(tokens)
        mask_s = [1 for i in range(l)]
        tokens = torch.LongTensor(tokens).unsqueeze(0).cuda()
        mask_s = torch.Tensor(mask_s).unsqueeze(0).cuda()
        # tokens = torch.LongTensor(tokens).unsqueeze(0)
        # mask_s = torch.Tensor(mask_s).unsqueeze(0)
        self.model.eval()
        logits = self.model((tokens,mask_s))
        pred = torch.argmax(logits, dim=1).item()
        return pred