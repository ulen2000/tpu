# encoding:utf-8

import os

import requests
import base64

'''
图像去雾
'''
import argparse
# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--file_path', help='image_path )', default='./images (2)_new.jpg', type=str)
args = parser.parse_args()

# 读取本地图片

filenamePath =args.file_path   # 测试图片存放在项目目录下
filename=os.path.basename(filenamePath).split('.')[0]


base64_data = ''
with open(filenamePath, "rb") as f:
    base64_data = base64.b64encode(f.read())

#request_url = "https://aip.baidubce.com/rest/2.0/image-process/v1/dehaze"#选一个
#request_url = "https://aip.baidubce.com/rest/2.0/image-process/v1/contrast_enhance"
#request_url = "https://aip.baidubce.com/rest/2.0/image-process/v1/image_definition_enhance"

params = {"image":base64_data}
access_token = '24.362ccd6961156082d1ae30d58a358214.2592000.1604643588.282335-227900481'#111
request_url = request_url + "?access_token=" + access_token
headers = {'content-type': 'application/x-www-form-urlencoded'}
response = requests.post(request_url, data=params, headers=headers)

if response:
    print ('response_success')
s = response.json()
imgdata=base64.b64decode(s['image'])
with open('%s_new.jpg'%(filename), "wb") as file:
    file.write(imgdata)
    file.close()
