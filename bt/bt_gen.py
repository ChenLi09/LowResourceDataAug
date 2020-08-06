#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File    :   bt_gen.py
@Time    :   2020/8/5
@Software:   PyCharm
@Author  :   Li Chen
@Desc    :   
"""

import http.client
import hashlib
import urllib
import random
import json
import time


def baidu_translate(client, ori_query, toLang='zh', fromLang='auto'):
    appid = '20200805000533734'
    secretKey = 'Rs9KEdIEoaAEZhUim0tA'
    time.sleep(1)
    salt = random.randint(32768, 65536)
    sign = appid + ori_query + str(salt) + secretKey
    sign = hashlib.md5(sign.encode()).hexdigest()
    myurl = '/api/trans/vip/translate'
    myurl += '?appid=' + appid + '&q=' + urllib.parse.quote(
        ori_query) + '&from=' + fromLang + '&to=' + toLang + '&salt=' + str(
        salt) + '&sign=' + sign
    new_query = []
    try:
        client.request('GET', myurl)
        response = client.getresponse()
        result_all = response.read().decode("utf-8")
        result = json.loads(result_all)
        for each in result['trans_result']:
            new_query.append(each['dst'])
    except Exception as e:
        print(e)
    return new_query


def back_translate(query):
    httpClient = http.client.HTTPConnection('api.fanyi.baidu.com')
    aug_query = [query]
    lan_list = "en,jp,kor,fra,spa,th,ara,ru,de".split(",")
    for tmp_lan in lan_list:
        for tmp_q in baidu_translate(httpClient, query, tmp_lan):
            aug_query.extend(baidu_translate(httpClient, tmp_q, 'zh'))
    httpClient.close()
    return aug_query


if __name__ == '__main__':
    result = back_translate('帮我查一下航班信息')
    print(result)
