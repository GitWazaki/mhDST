import os
import sys
import numpy as np
import random
import argparse
from shutil import copyfile
import sys
from os.path import abspath, join, dirname
sys.path.insert(0, join(abspath(dirname(__file__)), '../'))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from bert_model.utils import constant, helper
from bert_model.data.loader import DataLoader
from bert_model.model.trainer import MyTrainer
from sklearn import metrics


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='bert_model/dataset/88_clf', help='Dataset directory')
parser.add_argument('--bert_path', type=str, default='bert_model/bert-base-chinese', help='pretrained bert path')
parser.add_argument('--emb_dim', type=int, default=768, help='Word embedding dimension.')
parser.add_argument('--input_dropout', type=float, default=0.4, help='input dropout rate.')
parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Gradient clipping.')
parser.add_argument('--num_epoch', type=int, default=20, help='Number of total training epochs.')
parser.add_argument('--log_step', type=int, default=100, help='Print log every k steps.')
parser.add_argument('--batch_size', type=int, default=24, help='Training batch size.')
parser.add_argument('--save_dir', type=str, default='bert_model/saved_models', help='Root dir for saving models.')
parser.add_argument('--log', type=str, default='log.txt', help='File for loging the training process.')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
args = parser.parse_args()

# set random seed
'''
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(1234)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
'''

# make opt
opt = vars(args)
label2id = constant.ALL_LABEL_TO_ID
opt['num_class'] = len(label2id)
opt['special_tokens'] = constant.SPECIAL_TOKENS
# print opt
helper.print_config(opt)

# model save dir
helper.ensure_dir(opt['save_dir'], verbose=True)

# save config
helper.save_config(opt, opt['save_dir'] + '/config.json', verbose=True)
file_logger = helper.FileLogger(opt['save_dir'] + '/' + opt['log'], header="# epoch\ttrain_loss\ttest_loss\tP\tR\tF1")

# load data
print("Loading data from {} with batch size {} ...".format(opt['data_dir'], opt['batch_size']))
train_batch = DataLoader(opt['data_dir'] + '/train.tsv', opt['batch_size'], opt)
dev_batch = DataLoader(opt['data_dir'] + '/test.tsv', opt['batch_size'], opt)

print('Building model...')
trainer = MyTrainer(opt)

# start training
train_loss_his, test_loss_his, P_his, R_his, F1_his = [], [], [], [], []
for epoch in range(1, args.num_epoch+1):
    train_loss, train_step = 0., 0
    for i, batch in enumerate(train_batch):
        loss = trainer.update(batch)
        train_loss += loss
        train_step += 1
        if train_step % args.log_step == 0:
            print("train_loss: {}".format(train_loss/train_step))

    # eval on dev
    print("Evaluating on dev set...")
    test_loss, test_step = 0., 0
    right_num, logits_num, label_num = 1e-10, 1e-10, 1e-10
    preds = []
    labels = []
    for i, batch in enumerate(dev_batch):
        loss, pred, score,label = trainer.predict(batch)
        preds += pred
        labels += label
        test_loss += loss
        # right_num += tmp_r
        # logits_num += tmp_lo
        # label_num += tmp_la
        test_step += 1
        
    # P = right_num / logits_num
    # R = right_num / label_num
    # F1 = 2 * P * R / (P+R)
    P = metrics.precision_score(labels, preds, average='macro')
    R = metrics.recall_score(labels, preds, average='macro')
    F1 = 2 * P * R / (P+R)
    print("trian_loss: {}, test_loss: {}, P: {}, R: {}, F1: {}".format( \
        train_loss/train_step, test_loss/test_step, P, R, F1))
    file_logger.log("{}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}".format( \
        epoch, train_loss/train_step, test_loss/test_step, P, R, F1))
    target_names = list(label2id.keys())
    # target_names = ['无法办理值机','无法进行选座','无法修改选座','无法取消选座','为同行人值机失败','其他']
    # target_names = ['无法办理值机','无法进行选座','无法修改选座','无法取消选座','为同行人值机失败']
    print('report:')
    print(metrics.classification_report(labels, preds, target_names=target_names))
    train_loss_his.append(train_loss/train_step)
    test_loss_his.append(test_loss/test_step)
    P_his.append(P)
    R_his.append(R)

    # save
    model_file = opt['save_dir'] + '/checkpoint_epoch_{}.pt'.format(epoch)
    trainer.save(model_file)
    # save best model
    if epoch == 1 or F1 > max(F1_his):
        copyfile(model_file, opt['save_dir'] + '/best_model.pt')
        print("new best model saved.")
        print("")
        file_logger.log("new best model saved at epoch {}: {:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}"\
            .format(epoch, train_loss/train_step, test_loss/test_step, P, R, F1))
    F1_his.append(F1)
    print('max_f1:',max(F1_his))

print("Training ended with {} epochs.".format(epoch))
bt_train_loss = min(train_loss_his)
bt_F1 = max(F1_his)
bt_test_loss = min(test_loss_his)
print("best train_loss: {}, best test_loss: {}, best results: P: {}, R:{}, F1:{}".format( \
    bt_train_loss, bt_test_loss, P_his[F1_his.index(bt_F1)], R_his[F1_his.index(bt_F1)], bt_F1))
    #bt_train_loss, bt_test_loss, bt_P, R_his[P_his.index(bt_P)], F1_his[P_his.index(bt_P)]))
of = open('tmp.txt','a')
of.write(str(bt_train_loss)+","+str(bt_test_loss)+","+str(P_his[F1_his.index(bt_F1)])+str(R_his[F1_his.index(bt_F1)])+str(bt_F1)+'\n')
#of.write(str(bt_train_loss)+","+str(bt_test_loss)+","+str(bt_P)+str(R_his[P_his.index(bt_P)])+str(F1_his[P_his.index(bt_P)])+'\n')
of.close()
