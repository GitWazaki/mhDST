# 介绍
dialouge 是main文件，通过dialouge.reply(question)方法回答用户问题。stoires是现存的意图及其回答。
bert_model是意图识别模型相关代码，其中训练好的模型应放置在“bert_model/saved_models/”目录下，命名为"best_model.pt"。
app.py是一个利用dialouge开启后端接口的demo。