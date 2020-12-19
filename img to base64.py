import base64


import argparse
# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--file_path', help='image_path )', default='./11.jpg', type=str)
args = parser.parse_args()

# 读取本地图片
filenamePath =args.file_path   # 测试图片
base64_data = ''
with open(filenamePath, "rb") as f:
    base64_data = base64.b64encode(f.read())
    
requestBody = '{"dataArray":[{"name":"image","type":"stream","body":"'+base64_data.decode('ascii')+'"}]}'


#or
# -*- coding: utf-8 -*-
#!/usr/bin/python3
import base64
import commands

with open(r'11.jpg', 'rb') as f:
    ls_f=base64.b64encode(f.read()) #读取文件内容，转换为base64编码
    ls_f_new=ls_f.decode('utf-8')
