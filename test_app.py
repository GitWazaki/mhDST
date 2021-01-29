#!flask/bin/python
from flask import Flask
from flask import request
import json
from functools import lru_cache
from nlu.nlu import NLU
from dialouge import Dialouge
from flask_sqlalchemy import SQLAlchemy
import time
import pymysql
import logging

pymysql.install_as_MySQLdb()
app = Flask(__name__)

class Config(object):
    """配置参数"""
    # sqlalchemy的配置参数
    # 数据库类型: mysql
    # 用户名: root
    # 密码: root
    # 地址: 127.0.0.1
    # 端口: 3306
    # 数据库: test
    SQLALCHEMY_DATABASE_URI = "mysql://root:root@127.0.0.1:3306/hxDST"
    # 设置sqlalchemy自动跟踪数据库
    SQLALCHEMY_TRACE_MODIFICATIONS = True


app.config.from_object(Config) #添加配置
# db = SQLAlchemy(app) #初始化数据库（连接等操作）

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.WARNING)
app.logger.addHandler(stream_handler)
'''

CREATE TABLE `dialog` (
  `id` bigint(20) unsigned NOT NULL AUTO_INCREMENT,
  `sessionid` varchar(32) NOT NULL COMMENT '会话id',
  `reply` varchar(256) DEFAULT NULL COMMENT '会话回复',
  `intent` varchar(256) DEFAULT NULL COMMENT '目的',
  `states` varchar(256) DEFAULT NULL COMMENT '状态',
  `userinfo` varchar(2048) DEFAULT NULL COMMENT '用户信息',
  `msg` varchar(256) DEFAULT NULL COMMENT '会话内容',
  `pingjia` tinyint(1) DEFAULT '1' COMMENT '0好评1中评2差评,默认式中评',
  `ctime` varchar(20) NOT NULL COMMENT '创建时间',
  PRIMARY KEY (`id`)
) ENGINE=MyISAM AUTO_INCREMENT=1 DEFAULT CHARSET=utf8;

'''
'''
class Dialog(db.Model):
    __tablename__ = "dialog"
    id = db.Column(db.BigInteger,primary_key=True)
    sessionid = db.Column(db.String(32))
    reply =  db.Column(db.String(256))
    intent =  db.Column(db.String(256))
    states =  db.Column(db.String(256))
    userinfo =  db.Column(db.String(256))
    msg =  db.Column(db.String(256))
    pingjia =  db.Column(db.Integer)
    ctime =  db.Column(db.String(15))
    # 相当于__str__方法。
    def __repr__(self):
        return "dialog: %s %s" % (self.id,self.name)
'''
def addDialog(sid,reply,intent,states,userinfo,msg):
    now = time.time()  # 1s = 1000ms
    now = int(now * 1000)  # 获得13位时间戳

    data = Dialog()
    data.sessionid = sid
    data.reply = reply
    data.intent = intent
    data.states = json.dumps(states,ensure_ascii=False)
    data.userinfo = json.dumps(userinfo,ensure_ascii=False)
    data.msg = msg
    data.pingjia = 1
    data.ctime = str(now)
    db.session.add(data)
    # 最后插入完数据一定要提交
    db.session.commit()
    return data.id

#评价对话，传入id与评价值 pj 1好评2中评3差评
@app.route("/pj",methods=["POST"])
def pingjia():
    print(request.form)
    # id = request.args.get('id')
    id = request.form.get('id')
    # # id = request.form['id']
    id = int(id)
    # pj = request.args.get('pj')
    pj = request.form.get('pj')
    # # pj = request.form['pj']
    pj = int(pj)
    Dialog.query.filter_by(id=id).update({"pingjia": pj})
    db.session.commit()
    return_dict = {"return_code": 200, "return_info": "pingjia success"}
    return json.dumps(return_dict, ensure_ascii=False)


nlu = NLU()
@lru_cache(maxsize=10)
def get_chatbot(sid):
    return Dialouge(nlu)

@app.route("/chat",methods=["GET"])
def chat():
    return_dict= {'return_code': '200', 'return_info': '处理成功'}
    # 判断入参是否为空
    if request.args is None:
        return_dict['return_code'] = '201'
        return_dict['return_info'] = '请求参数为空'
        return json.dumps(return_dict, ensure_ascii=False)
    # 获取传入的params参数
    get_data=request.args.to_dict()
    message = get_data.get('message', '')
    sid = get_data.get('sessionid', -1)

    if sid == -1:
        return_dict = {
            "return_code": 202,
            "return_info": "无session id"
        }
    else:
        chatbot = get_chatbot(sid)
        text,user_info,intent,states = chatbot.reply(message)

        return_dict['input'] = message
        return_dict['text'] = text
        return_dict['intent'] = intent
        if return_dict['intent'] is None:
            return_dict['intent'] = '其他'
        return_dict['states'] = states
        return_dict['user_info'] = user_info
        #记录对话
        # id = addDialog(sid, text, intent, states, user_info, message)
        return_dict["id"] = id
    return json.dumps(return_dict, ensure_ascii=False)


if __name__ != '__main__':
	gunicorn_logger = logging.getLogger('gunicorn.error')
	app.logger.handlers = gunicorn_logger.handlers
	app.logger.setLevel(gunicorn_logger.level)

if __name__ == '__main__':
    # gunicorn_logger = logging.getLogger('gunicorn.error')
    # app.logger.handlers = gunicorn_logger.handlers
    # app.logger.setLevel(gunicorn_logger.level)
    app.run(debug=True,host='0.0.0.0',port =30018)