#coding:utf-8

import requests
import json

url = 'http://127.0.0.1:9000'
properties = {'annotators': 'tokenize,ssplit,pos,lemma,parse', 'outputFormat': 'json'}

# properties 要转成字符串, requests包装URL的时候貌似不支持嵌套的dict
params = {'properties' : str(properties)}

data = 'the brown fox'

resp = requests.post(url, data, params=params)
content=json.loads(resp.text)
for i in content['sentences']:
    print(i['parse'])

