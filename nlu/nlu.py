import os
import json
import logging
import jieba
import re
import pandas as pd

from nlu.intent_extractor import IntentExtractor
from nlu.userinfo_extractor import UserInfoExtractor

from bert_model import build_model
from bert_model.utils import constant

class NLU:
    def __init__(self):
        self.model = build_model.build()
        self.intent_list = list(constant.ALL_LABEL_TO_ID.keys())
        self.session = None
        self.state = dict()
        self.ticket_pattern = '(\\D)(\\d{13}|\\d{3}-\\d{10})(\\D)'
        self.flight_pattern = '(\\D)((([A-Za-z][A-Za-z\\d])|([A-Za-z\\d][A-Za-z]))\\d{3,4})(\\D)'
        self.id_pattern = '(\\D)(\\d{18}|\\d{17}[xX])(\\D)'
        self.date_pattern = '(\\D)((\\d{1,2}[月.])?\\d{1,2}[日号])(\\D)'
        self.tel_pattern = '(\\D)((13[0-9]|14[5-9]|15[0-3,5-9]|16[2,5,6,7]|17[0-8]|18[0-9]|19[0-3,5-9])\\d{8})(\\D)'
        self.seat_pattern = '(\\D)(\\d{2}[A-Za-z])(\\D)'
        self.user_info_lists = json.load(open('nlu/user_info.json',encoding="utf-8"))
        self.intent_extractor = IntentExtractor()
        # self.userinfo_extractor = UserInfoExtractor()
    
    def construct_dialogue(self, text):
        text = re.sub(self.ticket_pattern, '\\g<1>[ticket]\\g<3>', " "+text+" ").strip()
        text = re.sub(self.flight_pattern, '\\g<1>[flight]\\g<3>', " "+text+ " ").strip()
        text = re.sub(self.id_pattern, '\\g<1>[id]\\g<3>', " "+text+ " ").strip()
        text = re.sub(self.date_pattern, '\\g<1>[date]\\g<3>', " "+text+ " ").strip()
        text = re.sub(self.tel_pattern, '\\g<1>[tel]\\g<3>', " "+text+ " ").strip()
        text = re.sub(self.seat_pattern,'\\g<1>[seat]\\g<3>'," "+text+" ").strip()
        return text

    def extract_num_slots(self, sentence):
        num_slots = list()
        ticket = re.search(self.ticket_pattern, " "+sentence+" ")
        flight = re.search(self.flight_pattern, " "+sentence+" ")
        id = re.search(self.id_pattern, " "+sentence+" ")
        date = re.search(self.date_pattern, " "+sentence+" ")
        tel = re.search(self.tel_pattern, " "+sentence+" ")
        seat = re.search(self.seat_pattern, " "+sentence+" ")
        for re_obj, name in zip([ticket, flight, id, date, tel, seat], ['票号', '航班号', '身份证', '航班日期', '电话', '座位号']):
            if re_obj:
                num_slots.append((name, re_obj.group(2)))
        return num_slots

    def check_list(self,lis,text):
        return any([ele in text for ele in lis])

    def check_intent(self,text,intent):
        # 操作失败型
        if self.check_list(['无法','失败','不能'],text) and self.check_list(['座位','选座','值机'],text):
            if self.check_list(['朋友','同行人'],text):
                return '为同行人值机失败'
            if self.check_list(['修改','改','换'],text):
                return '无法修改选座'
            if self.check_list(['取消','退'],text):
                return '无法取消选座'
            if self.check_list(['选座','座','坐'],text):
                return '无法办理值机选座'
            elif intent != '为同行人值机失败' and self.check_list(['值机'],text):
                return '无法办理值机选座'
        if self.check_list(['查看'],text) and self.check_list(['登机牌'],text):
                return '如何查看登机牌'
        return intent

    def check_text_userinfo(self,key_list,text):
        for key in key_list:
            if isinstance(key,str) and key not in text:
                return False
            if isinstance(key,list) and not any([ele for ele in key if ele in text]):
                return False
        return True

    def extract_userinfo_by_rule(self,text):
        user_info = {}
        for name in self.user_info_lists:
            for key in self.user_info_lists[name]:
                if self.check_text_userinfo(self.user_info_lists[name][key],text):
                    user_info[name] = key
        return user_info

    def extract_intent_by_rule(self, text):
        intent = self.intent_extractor.extract(text)
        return intent
    '''
    def extract_userinfo_by_rule(self, text):
        userinfo = self.userinfo_extractor.extract(text)
        return userinfo
    '''
    def parse(self, sentence, human_utterance=True):
        text = self.construct_dialogue(sentence)
        current_state = self.extract_num_slots(sentence)
        userinfo = self.extract_userinfo_by_rule(text)
        intent = self.extract_intent_by_rule(text)
        if intent is None:
            pred = self.model.run_pred_single(text)
            intent = self.intent_list[pred]

        # intent = self.check_intent(text,intent)
        return intent, pred, current_state, userinfo, text

if __name__ == '__main__':
    nlu = NLU()
    print(nlu.parse('证件号231121199606204633'))
    print(nlu.parse('证件号'))
    print(nlu.parse('无法进行选座'))
    print(nlu.parse('无法修改选座'))
    print(nlu.parse('无法取消选座'))
    print(nlu.parse('为同行人值机失败'))
    print(nlu.parse('票号是7818298493048'))
    print(nlu.parse('票号'))

