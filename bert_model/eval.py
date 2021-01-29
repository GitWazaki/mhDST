import os
import sys
import numpy as np
import random
import argparse
from shutil import copyfile
from sklearn import metrics
from sklearn.metrics import confusion_matrix
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


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='bert_model/dataset/88_clf', help='Dataset directory')
parser.add_argument('--save_dir', type=str, default='bert_model/saved_models', help='Root dir for saving models.')
args = parser.parse_args()

# make opt
opt = vars(args)
opt = helper.load_config(opt['save_dir'] + '/config.json', verbose=True)
# load data
print("Loading data from {} with batch size {} ...".format(opt['data_dir'], opt['batch_size']))
dev_batch = DataLoader(opt['data_dir']+'/test.tsv', opt['batch_size'], opt)
raw_data = [dev_batch.raw_data[i:i+opt['batch_size']] for i in range(0, len(dev_batch.raw_data), opt['batch_size'])]
tmp = []
for db in raw_data:
    indexes = list(range(len(db)))
    db = sorted(zip(db, indexes), key=lambda x:(len(x[0].split('\t')[0]),x[1]), reverse=True)
    tmp += ([bes[0] for bes in db])
raw_data = tmp


print('Building model...')
trainer = MyTrainer(opt)
model_file = os.path.join(opt['save_dir'], 'best_model.pt')
trainer.load(model_file)

# eval on dev
print("Evaluating on dev set...")
test_loss, test_step = 0., 0
right_num, logits_num, label_num = 0., 0., 0.
labels = []
preds = []
scores = []
for i, batch in enumerate(dev_batch):
    loss, pred,score,label = trainer.predict(batch)
    preds += pred
    labels += label
    scores += score
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

target_names = list(constant.ALL_LABEL_TO_ID.keys())
# target_names = ['无法办理值机','无法进行选座','无法修改选座','无法取消选座','为同行人值机失败','其他']
# target_names = ['无法办理值机','无法进行选座','无法修改选座','无法取消选座','为同行人值机失败']
print('report:')
print(metrics.classification_report(labels, preds, target_names=target_names))
labels = [target_names[ele] for ele in labels]
preds = [target_names[ele] for ele in preds]
print(confusion_matrix(labels,preds,labels=target_names))
# write texts
# tmp = []
# for batch in raw_data:
#     tmp += batch
with open('pred_results.txt','w') as fout:
    fout.write('text\tpred\tlabel\n')
    for raw,pred,label,score in zip(dev_batch.raw_data,preds,labels,scores):
        text = raw.split('\t')[0]
        fout.write(text + '\t' + str(pred) +'\t' + str(label) + '\t' +str(score) + '\n')
with open('error_list.txt','w') as fout:
    fout.write('text\tpred\tlabel\n')
    for raw,pred,label,score in zip(dev_batch.raw_data,preds,labels,scores):
        text = raw.split('\t')[0]
        if pred != label:
            fout.write(text + '\t' + str(pred) +'\t' + str(label) + '\t' + str(score) + '\n')
# open("error_analysis.txt", 'w').write(str(tmp))
print("test_loss: {}, P: {}, R: {}, F1: {}".format(test_loss/test_step, P, R, F1))

                


'''
def predict(sentences):
    test_data = list()
    for sent in sentences:
        tokens = jieba.lcut(sent, cut_all=False)
        test_data.append({'text': tokens, 'aspects': [constant.ID_TO_ASP[0]], 'polarities': [constant.ID_TO_LABEL[0]]})
    test_batch = DataLoader(test_data, opt['batch_size'], opt, vocab)
    print("Predicting on test set...")
    labels = list()
    for i, (batch, indices) in enumerate(test_batch):
        predicts = trainer.predict(batch)
        labels += [predicts[k] for k in indices]
    results = list()
    for i, label in enumerate(labels):
        aspects = [x1 for x1, x2 in label]
        polarities = [x2 for x1, x2 in label]
        results.append({'text': test_data[i]['text'], 'aspects': aspects, 'polarities': polarities})
    return results

if __name__ == '__main__':
    sentences = eval(open('test_sentences.txt', 'r', encoding='utf-8').read())
    results = predict(sentences)
    open('results.txt', 'w', encoding='utf-8').write(str(results))
''' 
