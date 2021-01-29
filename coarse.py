import json


class Coarse(object):

    def __init__(self, name: str, intent_dict: dict, action_dict: dict, intent_id_dict: dict):
        self.name = name
        self.intent_dict = intent_dict
        self.action_dict = action_dict
        self.intent_id_dict = intent_id_dict
        # if not isinstance(intent_list,list):
        # 	raise ValueError('wrong intentlist format')

    def matchCoarse(self,intent):
        return self.name == intent

    def matchFine(self,text):
        # print(text)
        # numList = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        # if text in numList :
        #     text = int(text)
        if text in self.action_dict.keys():
            return True, self.intent_dict[text], self.action_dict[text]
        else:
            return False, None, None

    def getCoarseByIntent(self,intent):
        if intent in self.intent_dict.values():
            return True, self.name
        return False, None

    def get_intent_id(self,text):
        return self.intent_id_dict[text]


class CoarseReader(object):

    @staticmethod
    def convert_from_json(data):
        name = data.get("name","story")
        intent = data.get("intent",None)
        action = data.get("action",None)
        intent_id = data.get("intent_id",None)
        coarse = Coarse(name, intent, action, intent_id)
        return coarse


    @staticmethod
    def read_json(file):
        with open(file,encoding="utf-8") as f:
            datas = json.load(f)
        coarses = []
        for data in datas['coarses']:
            coarses.append(CoarseReader.convert_from_json(data))
        return coarses

if __name__ == '__main__':
    coarses = CoarseReader.read_json('coarses.json')
    for coarse in coarses:
        print(coarse.name,coarse.intent_dict,coarse.action_dict,coarse.intent_id_dict)
        print(coarse.matchCoarse("乘机流程问题_出发流程"))
        print(coarse.matchFine("1"))