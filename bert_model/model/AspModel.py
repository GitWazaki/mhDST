import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from transformers import BertTokenizer, BertModel

from bert_model.utils import constant

def get_long_tensor(tokens_list, batch_size):
    """ Convert list of list of tokens to a padded LongTensor. """
    token_len = max(len(x) for x in tokens_list)
    tokens = torch.LongTensor(batch_size, token_len).fill_(constant.PAD_ID)
    for i, s in enumerate(tokens_list):
        tokens[i, :len(s)] = torch.LongTensor(s)
    return tokens

def get_float_tensor(tokens_list, batch_size):
    """ Convert list of list of tokens to a padded FloatTensor. """
    token_len = max(len(x) for x in tokens_list)
    tokens = torch.FloatTensor(batch_size, token_len).fill_(0)
    for i, s in enumerate(tokens_list):
        tokens[i, :len(s)] = torch.FloatTensor(s)
    return tokens

def get_asp_tokens(opt):
    tokenizer = BertTokenizer.from_pretrained(opt['bert_path'])
    asps = list(constant.ASP_TO_ID.keys())
    asps = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(asp)) for asp in asps]
    mask_asp = []
    for asp in asps:
        mask_asp.append([1 for i in asp])
    asps = get_long_tensor(asps, len(asps))
    mask_asp = get_float_tensor(mask_asp, len(asps))
    
    return asps.cuda(), mask_asp.cuda()
    

class AspModel(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.bert = BertModel.from_pretrained(self.opt['bert_path']).cuda()
        # self.bert = BertModel.from_pretrained(self.opt['bert_path'])
        # fine-tuning the bert parameters
        for param in self.bert.parameters():
            param.requires_grad = True
        self.input_dropout = nn.Dropout(opt['input_dropout'])
        # asp_embs, text_cls = self.bert(asps, attention_mask=mask_asp)
        # self.asp_embs = asp_embs.sum(dim=1)/mask_asp.sum(dim=1).unsqueeze(-1)
        # self.asp_emb = nn.Embedding(len(constant.ASP_TO_ID), opt['emb_dim'])
        # self.asp_emb.weight.data.copy_(self.asp_embs)

        self.classifier = nn.Linear(opt['emb_dim'], opt['num_class'])

    def forward(self, inputs):
        # unpack inputs 
        tokens, mask_s = inputs

        # ipnut embs
        encoder_out, text_cls = self.bert(tokens, attention_mask=mask_s)
        text_cls = self.input_dropout(text_cls)
        # # attention
        # att_q = self.asp_emb.weight.unsqueeze(0).repeat(encoder_out.size(0), 1, 1).transpose(1, 2)
        # att_m = encoder_out.bmm(att_q)                          # [batch_size, len, asp_num]
        # mask_s = mask_s.unsqueeze(-1).repeat(1, 1, att_m.size(2))
        # att_m = torch.where(mask_s==0, torch.zeros_like(att_m)-10e10, att_m)
        # att_m = F.softmax(att_m, dim=1).transpose(1, 2)         # [batch_size, asp_num, len]
        # c_inputs = att_m.bmm(encoder_out)                       # [batch_size, asp_num, rnn_hidden*2]
        
        # logits 
        logits = self.classifier(text_cls)
        
        return logits
