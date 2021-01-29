import re
import os
from .rule_parser import rule_to_regex

class IntentExtractor:
    def __init__(self):
        # [(regex, intent), ...]
        self.rules = self.read_rules()

    def read_rules(self):
        rules = []
        intent = None

        file = os.path.join(os.path.dirname(__file__), 'intent_rules.txt')

        with open(file, mode='r', encoding='utf-8') as fin:
            for line in fin:
                line = line.strip()

                # comment
                if line.startswith('//'):
                    continue
                
                # intent
                if line.startswith("##"):
                    intent = line.lstrip('# ')

                # rule
                if line.startswith('- '):
                    if intent is None:
                        raise ValueError("rule file format error")
                    rule = line[2:]
                    rule = rule.strip(' ')
                    regex = rule_to_regex(rule)
                    rules.append((regex, intent))
                    
        return rules                    
                    

    def extract(self, text):
        for rule in self.rules:
            if re.search(rule[0], text) is not None:
                return rule[1]


if __name__ == "__main__":
    intent_extractor = IntentExtractor()
    intent = intent_extractor.extract("显示已经完成值机，但是我没有值机呀")
    print(intent) # 显示完成值机