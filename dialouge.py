import json

from coarse import CoarseReader
from nlu.nlu import NLU
import datetime
import re
from story import StoryReader
# nlu = NLU()
class Dialouge:

    def __init__(self,nlu,sid):
        self.global_intent = None
        self.states = {}
        self.text_user_info = {}
        self.intent_his = ['其他']
        self.times = 0
        self.nlu = nlu
        self.not_find_error = '抱歉，暂时无法解决您的问题'
        self.error_intent = ['']
        self.last_text = ''
        self.sid = sid
        self.stories = StoryReader.read_json('stories.json')
        self.coarses = CoarseReader.read_json('coarses.json')
        self.coarses_list = ['乘机流程问题_出发流程','乘机流程问题_到达流程','乘机流程问题_乘机证件问题','乘机流程问题_乘机注意事项','乘机流程问题_特殊人群','航班问题_延误或取消后处理方法','航班问题_航班退票改签无航司','票务问题_优惠机票','票务问题_婴幼儿购票','行李问题_行李托运规则','行李问题_携带物品要求_可随身携带物品要求']
        self.history_list = []

    def get_last_text(self):
        return self.last_text

    def deal_others(self, intent,text):
        if intent in self.general_reply:
            return self.general_reply[intent]
        return '抱歉,暂时无法回答'

    def current_intent(self):
        # 尝试通过history list 判断current_intent
        if self.history_list:
            last_intent = self.history_list[-1]['parseInfo']['intent']
            print("last intent", last_intent)
            intent = self.tryGetCoarse(last_intent)
            if not intent:
                intent = self.intent_his[-1]
            if intent == '其他':
                intent = self.global_intent
        # intent = self.intent_his[-1]
        # if intent == '其他':
        #     return self.global_intent
            return intent
        return None

    def update_intent(self,intent,text):
        if self.global_intent is None and intent == '其他':
            if self.intent_his[-1] != '其他':
                intent,_,_,_ = self.nlu.parse(self.intent_his[-1] + ','+text)
        self.intent_his.append(intent)
        return intent

    def tryGetCoarse(self,intent):
        if intent in self.coarses_list:
            return intent
        for coarse in self.coarses:
            judge, cname = coarse.getCoarseByIntent(intent)
            if judge:
                return cname
        return None

    def get_groupID(self,intent):
        env = intent.split('_')[0]
        if env == '航班问题':
            return 1
        elif env == '行李问题':
            return 2
        elif env == '票务问题':
            return 3
        elif env == '乘机流程问题':
            return 4
        else:
            return 0

    def policy(self, intent):
        if self.global_intent is not None:
            self.times += 1
        # current_intent = self.current_intent()
        for story in self.stories:
            if story.match(intent, self.text_user_info):
                return story.act(self)
        return self.not_find_error


    def reply(self, text):
        print('text',text)
        input_para = {'content': text, 'mode': "text"}
        current_intent = self.current_intent()
        print('current_intent',current_intent)
        # 处理粗粒度
        for coarse in self.coarses:
            if coarse.matchCoarse(current_intent):
                judge, intent, action=coarse.matchFine(text)
                print(judge,intent,action)
                # intent = self.update_intent(intent, text)
                if judge:
                    input_para['user_info'] = None
                    input_para['history_list'] = self.history_list
                    return 0,action, self.text_user_info, intent, coarse.get_intent_id(text), self.get_groupID(intent), self.states, input_para

        intent, intent_id, states, text_user_info, processed_text = self.nlu.parse(text)
        if intent == '其他' or intent == '负例':
            if current_intent == '行李问题_行李托运规则':
                text = '托运'+text
            elif current_intent == '乘机流程问题_出发流程':
                text = text+'出发流程'
            elif current_intent == '乘机流程问题_到达流程':
                text = text+'到达流程'
            elif current_intent == '乘机流程问题_特殊人群':
                text = text + '病残旅客'
            elif current_intent == '航班问题_航班退票改签无航司':
                text = text + '退票'
            print("in 负例/其他",text)
            intent, intent_id,states,text_user_info, processed_text = self.nlu.parse(text)

        intent = self.update_intent(intent,text)

        input_para['user_info'] = text_user_info
        input_para['history_list'] = self.history_list
        print("ans:",intent,self.policy(intent))
        # action
        return 1,self.policy(intent), text_user_info, intent, intent_id, self.get_groupID(intent),states, input_para

    def addRound(self, round):
        self.history_list.append(round)

if __name__ == '__main__':
    nlu = NLU()
    dialouge = Dialouge(nlu,1)
    print('question:',"飞机起飞桌椅要调直吗")
    print('reply:',dialouge.reply('飞机起飞桌椅要调直吗'))