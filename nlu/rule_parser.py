import re
from typing import List

# 目前支持下面这些 token 类型
TYPE_MAP = {
    '*',
    '|',
    '(',
    ')',
    '[',
    ']',
    'WORD'
}

class Token:
    def __init__(self, type, value=''):
        self.type = type
        self.value = value

    def __str__(self):
        if not self.value:
            return '<{}>'.format(self.type)
        else:
            return '<{}, {}>'.format(self.type, self.value)
    

class Lex:
    def __init__(self, text):
        self.text = text
        self.index = 0
    
    def next_token(self):
        while self.lookahead(0) != '':
            ch = self.lookahead(0)
            if ch in '[]()*|':
                self.consume(ch)
                return Token(ch)
            else:
                return self.word()
        return Token('EOF')
        
    def word(self):
        STOP_WORDS = {'(', ')', '[', ']', '|'}
        word = ''

        ch = self.lookahead(0)
        while ch != '':
            if ch in STOP_WORDS:
                break
            if ch == '\\' and self.lookahead(1) in STOP_WORDS:
                word += ch
                self.consume(ch)
                ch = self.lookahead(0)
                word += ch
                self.consume(ch)
            else:
                word += ch
                self.consume(ch)

            ch = self.lookahead(0)
        
        return Token('WORD', word)
            

    def lookahead(self, i):
        if self.index + i < len(self.text):
            return self.text[self.index + i]
        else:
            return ''

    def consume(self, ch):
        if self.index >= len(self.text):
            raise EOFError("no content available for consuming")
        if self.text[self.index] != ch:
            raise ValueError("consume char unmatch, intent: {}  actual: {}".format(ch, self.text[self.index]))
        self.index += 1

    def tokens(self):
        tokens = []

        token = self.next_token()
        while token.type != 'EOF':
            tokens.append(token)
            token = self.next_token()

        return tokens


class Parser:
    def __init__(self, text):
        self.text = text
        lex = Lex(text)
        self.tokens = lex.tokens()

    def convert_to_regex(self):
        vaild = self.check_vaild(self.tokens)
        if vaild == False:
            raise ValueError("template is invaild!")
        
        
        regex = ''

        for token in self.tokens:
            # print(token)
            type = token.type

            if type == '*':
                regex += '.*?'
            elif type == '|':
                regex += '|'
            elif type == '[' or type == '(':
                regex += '('
            elif type == ']':
                regex += ')?'
            elif type == ')':
                regex += ')'
            elif type == 'WORD':
                regex += self.normalize(token.value)
            else:
                raise ValueError("unexcept token type: {}".format(type))
        
        return regex

    def check_vaild(self, tokens: List[Token]):
        stack = []

        start = '[('
        end = '])'
        mapping = {
            end[i]: start[i] for i in range(len(start))
        }
        
        for token in tokens:
            ch = token.type
            if ch in start:
                stack.append(ch)
            elif ch in end:
                if len(stack) > 0 and stack[-1] == mapping[ch]:
                    stack.pop()
                else:
                    return False
            else:
                pass
        
        return len(stack) == 0

    def normalize(self, text):
        # 某些和正则表达式冲突的字符可能需要转义
        special_chars = r'^$?.*'

        special_chars = ''.join(['\\' + ch for ch in special_chars])

        text = re.sub('([' + special_chars  + '])', r'\\\1', text) 

        return text


def rule_to_regex(template):
    parser = Parser(template)
    return parser.convert_to_regex()

def match(template, text):
    regex = rule_to_regex(template)
    return re.search(regex, text) != None


if __name__ == "__main__":
    template = "*[请问[一下]](北京)[的](天.气|气温)[怎么样|如何]"

    parser = Parser(template)
    assert match(template, '你好，请问北京的气温如何') == True
    print(parser.convert_to_regex())