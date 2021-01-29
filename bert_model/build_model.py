import os
import sys
import numpy as np
import random
import argparse
from shutil import copyfile

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


from bert_model.utils import constant, helper
from bert_model.model.trainer import MyTrainer

def build():
    # args = {'save_dir':'bert_model/saved_models'}
    # args['save_dir'] = 'bert_model/saved_models'
    # make opt
    opt = vars()
    opt['save_dir'] = 'bert_model/saved_models'
    opt['sepcial_tokens'] = constant.SPECIAL_TOKENS
    opt = helper.load_config(opt['save_dir'] + '/config.json', verbose=True)
    print('Building model...')
    trainer = MyTrainer(opt)
    model_file = os.path.join(opt['save_dir'], 'best_model.pt')
    print('model file',model_file)
    trainer.load(model_file)
    return trainer