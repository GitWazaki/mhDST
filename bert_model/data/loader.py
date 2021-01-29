"""
Data loader for nyt json files.
"""

import json
import random
import torch
import numpy as np
from transformers import BertTokenizer
import random
from bert_model.utils import constant

# get word piece tokenizer 
def get_tokenizer(bert_path):
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    return tokenizer

class DataLoader(object):
    """
    Load data from files, preprocess and prepare batches.
    """
    def __init__(self, filename, batch_size, opt):
        self.batch_size = batch_size
        self.opt = opt
        self.tokenizer = get_tokenizer(opt['bert_path'])
        self.tokenizer.add_special_tokens({'additional_special_tokens':opt['special_tokens']})
        self.label2id = constant.ALL_LABEL_TO_ID
        # self.asp2id = constant.ASP_TO_ID
        # load samples
        # data = eval(open(filename, 'r',encoding='utf-8').read())
        data = open(filename,'r',encoding='utf-8').readlines()
        random.shuffle(data)
        self.raw_data = data
        # preprocess sentences 
        data = self.preprocess(data, opt)
        self.num_examples = len(data)
        # chunk into batches
        data = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
        self.data = data
        print("{} batches created for {}".format(len(data), filename))
     
    def preprocess(self, data, opt):
        """ Preprocess the data and convert to ids. """
        processed = []
        for d in data:
            # tokenize 
            d = d.strip().split('\t')
            text = d[0]
            try:
                label = d[1]
            except Exception as e:
                print(d)
            label = self.label2id[label]
            # print('text:',text,'label:',label)
            # text = ' '.join(d['text'])
            tokens = self.tokenizer.tokenize(text)
            tokens = ['[CLS]']+tokens+['[SEP]']
            
            # labels
            # label = [self.label2id['-2']] * len(self.asp2id)
            # for asp, pol in zip(d['aspects'], d['polarities']):
            #     if asp in self.asp2id.keys():
            #         label[self.asp2id[asp]] = self.label2id[pol]
            # # delete the non-label sentences
            # if label.count(self.label2id['-2']) == len(self.asp2id):
            #     continue
                
            # mapping to ids
            tokens = self.tokenizer.convert_tokens_to_ids(tokens)
            # delete the 0 length sentences 
            l = len(tokens)
            if l == 0:
                continue
                
            # mask for attention 
            mask_s = [1 for i in range(l)]
            processed += [(tokens, mask_s, int(label))]
            
        return processed

    def __len__(self):
        return len(self.data)

    # 0: tokens, 1: mask_s, 2: label
    def __getitem__(self, key):
        """ Get a batch with index. """
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError
        batch = self.data[key]
        batch_size = len(batch)
        batch = list(zip(*batch))
        assert len(batch) == 3
        
        # sort all fields by lens for easy RNN operations
        lens = [len(x) for x in batch[0]]
        # batch, _ = sort_all(batch, lens)

        # convert to tensors
        words = get_long_tensor(batch[0], batch_size)
        
        # mask_s to tensors
        mask_s = get_float_tensor(batch[1], batch_size)

        # label to tensors
        label = torch.LongTensor(batch[2])

        return (words, mask_s, label)

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

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

def map_to_ids(tokens, vocab):
    ids = [vocab[t] if t in vocab else constant.UNK_ID for t in tokens]
    return ids

def sort_all(batch, lens):
    """ Sort all fields by descending order of lens, and return the original indices. """
    unsorted_all = [lens] + [range(len(lens))] + list(batch)
    sorted_all = [list(t) for t in zip(*sorted(zip(*unsorted_all), reverse=True))]
    return sorted_all[2:], sorted_all[1]

def word_dropout(tokens, dropout):
    """ Randomly dropout tokens (IDs) and replace them with <UNK> tokens. """
    return [constant.UNK_ID if x != constant.UNK_ID and np.random.random() < dropout \
            else x for x in tokens]

