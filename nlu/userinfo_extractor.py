import re
import os
from .rule_parser import rule_to_regex

class UserInfoExtractor:
    def __init__(self):
        # [(regex, intent), ...]
        self.rules = self.read_rules()

    def read_rules(self):
        rules = []
        slot = None
        value = None
        file = os.path.join(os.path.dirname(__file__), 'intent_rules.txt')

        with open(file, mode='r', encoding='utf-8') as fin:
            for line in fin:
                line = line.strip()

                # comment
                if line.startswith('//'):
                    continue
                
                # intent
                if line.startswith("##"):
                    line = line.lstrip('# ')
                    line = line.split(':')
                    if len(line) < 2:
                        raise ValueError("rule file format error")
                    slot = line[0]
                    value = line[1]
                # rule
                if line.startswith('- '):
                    if slot is None or value is None:
                        raise ValueError("rule file format error")
                    rule = line[2:]
                    rule = rule.strip(' ')
                    regex = rule_to_regex(rule)
                    rules.append((regex, {slot:value}))
                    
        return rules                    
                    

    def extract(self, text):
        userinfo = {}
        for rule in self.rules:
            if re.search(rule[0], text) is not None:
                userinfo.update(rule[1])
        return userinfo


if __name__ == "__main__":
    intent_extractor = IntentExtractor()
    userinfo = intent_extractor.extract("显示已经完成值机，但是我没有值机呀")
    print(userinfo) # 显示完成值机