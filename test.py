import requests


turn_lis = [
[("充电宝能带上飞机吗？","行李问题_携带物品要求_充电宝")]
]

qa_url = "http://192.168.5.201:30018/chat"

for turn_id,turn in enumerate(turn_lis):
	for ele in turn:
		text = ele[0]
		intent = ele[1]
		res = requests.get(qa_url,params={'message':text,'sessionid':str(turn_id)})
		res = res.json()
		if res['return_code'] != '200':
			print('request error',text)
		if res['intent'] != intent:
			print('intent error',text)
			print('pred intent',res['intent'])
			print('true intent',intent)
