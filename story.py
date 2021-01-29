import json
class Story(object):

	def __init__(self, name: str, intent_list: list, userinfo: dict, action_list: list):
		'''
		intent:需要匹配用户意图列表
		userinfo: 用户信息，字典, 每一项的值为(value,bool)对，代表匹配或者不匹配
		check_true: 匹配还是不匹配
		action：动作列表，每一项也是列表，其中第一项为函数，后续为函数参数参数
		'''
		self.name = name
		self.intent_list = intent_list
		if not isinstance(intent_list,list):
			raise ValueError('wrong intentlist format')
		self.userinfo = userinfo
		self.action_list = action_list

	def match(self,intent,userinfo):
		if all([intent != ele for ele in self.intent_list]):
			return False
		for key in self.userinfo:
			if key not in userinfo:
				return False
			value, check_true= self.userinfo[key]
			if check_true:
				if userinfo[key] != value:
					return False
			elif userinfo[key] == value:
				return False
		return True

	def act(self,dm):
		for action in self.action_list:
			act_name = action[0]
			if act_name == 'reply':
				return action[1]
			if act_name == 'ask_userinfo':
				return dm.ask_userinfo(dm.current_intent(),dm.get_last_text())
			return 'action not defined'


class StoryReader(object):

	@staticmethod
	def convert_from_json(data):
		name = data.get("name","story")
		intent = data.get("intent",None)
		userinfo = data.get("userinfo",None)
		action = data.get("action",None)
		story = Story(name, intent, userinfo, action)
		return story

	@staticmethod
	def read_json(file):
		with open(file,encoding="utf-8") as f:
			datas = json.load(f)
		stories = []
		for data in datas['stories']:
			stories.append(StoryReader.convert_from_json(data))
		return stories

if __name__ == '__main__':
	stories = StoryReader.read_json('stories.json')
	for story in stories:
		print(story.name,story.intent_list,story.userinfo,story.action_list)
		print(story.match('无法办理值机',{'用户旅客身份':'VIP'}))