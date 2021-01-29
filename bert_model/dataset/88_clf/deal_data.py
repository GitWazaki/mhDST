import random
from collections import defaultdict

datas = []
labels = defaultdict(list)
with open('all_data.tsv',encoding='utf-8') as f:
	for line in f:
		split_line = line.strip().split('\t')
		labels[split_line[-1]].append(line)

lines = []
for lis in labels.values():
	if len(lis) > 1:
		lines += lis

with open('all_data_filtered.tsv','w',encoding='utf-8') as fout:
	for line in lines:
		fout.write(line)



random.shuffle(lines)
ll = len(lines)
train_ll = int(ll * 0.8)
train_data = lines[:train_ll]
test_data = lines[train_ll:]
with open('train.tsv','w',encoding='utf-8') as fout:
	for line in train_data:
		fout.write(line)

with open('test.tsv','w',encoding='utf-8') as fout:
	for line in test_data:
		fout.write(line)