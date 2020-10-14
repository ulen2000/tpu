# encoding:utf-8
import time
import urllib.request as urllib2
import base64

import requests

from aip import AipImageProcess
import os

import requests
import base64
import argparse



'''
easydl图像分类
'''


# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--file', help='image_path )', default='./U080-000001.png', type=str)
args = parser.parse_args()
flag=''
value=0
time1=time.time()
# 读取本地图片
filenamePath =args.file   # 测试图片存放在项目目录下
father_path = os.path.dirname(filenamePath)+ os.path.sep
filename=os.path.basename(filenamePath).split('.')[0]

base64_data = ''
with open(filenamePath, "rb") as f:
    image=f.read()
    base64_data = base64.b64encode(image)
time2=time.time()
print('read image:%s'%(time2-time1))
request_url = "https://aip.baidubce.com/rpc/2.0/ai_custom_pro/v1/classification/weather"
#{'refresh_token': '25.64233fa67dd30c69a5cfee559c053eea.315360000.1917406344.282335-22789811',
# 'expires_in': 2592000,
# 'session_key': '9mzdCyCSK2/B99CmauQpVXgu5rzyerdkhKJJwli7o+txk+vn7OTlPnVZaIo4bEBGHzvLNSsEmhNwYyBELwvzYoaZzhQkdw==',

# 'access_token': '24.08c996e559bbb0729299622c734f4413.2592000.1604638344.282335-22789811',

# 'scope': 'public brain_all_scope brain_object_detect easydl_mgr easydl_retail_mgr ai_custom_retail_image_stitch ai_custom_test_oversea easydl_pro_mgr wise_adapt lebo_resource_base
# lightservice_public hetu_basic lightcms_map_poi kaidian_kaidian ApsMisTest_Test权限 vis-classify_flower lpq_开放 cop_helloScope ApsMis_fangdi_permission smartapp_snsapi_base iop_autocar
# oauth_tp_app smartapp_smart_game_openapi oauth_sessionkey smartapp_swanid_verify smartapp_opensource_openapi smartapp_opensource_recapi fake_face_detect_开放Scope vis-ocr_虚拟人物助理 idl-video_虚拟人物助理 smartapp_component',
# 'session_secret': '95c507bbdb22cd0ac04b958b79eca00e'}

params = '{\"image\":\"'+base64_data.decode('ascii')+'\",\"top_num\":\"5\"}'

access_token = '24.08c996e559bbb0729299622c734f4413.2592000.1604638344.282335-227898111'#111
request_url = request_url + "?access_token=" + access_token

response = requests.post(url=request_url, data=params)
time3=time.time()
print('transfer and process time：%s'%(time3-time2))
s = response.json()
print('total time:%s'%(time3-time1))

list=s['results']
for i in list:
    print("序号：%s   值：%s" % (list.index(i) + 1, i)) #i={'name': 'foggy', 'score': 0.9983261227607727}
    if i['score']>value:
        value=i['score']
        flag = i['name']
print('so the highest possibility is: ' +flag +', score: ' + str(value))
print('-------------------start preprocess-------------------')

""" 你的 APPID AK SK """
APP_ID = '22790048'
API_KEY = '6pAtaurlGDbiEtDanrVTQBal'
SECRET_KEY = 'aYAHS5BOKI5gTPGgdzz0mu2bGMUmwtNV'

client = AipImageProcess(APP_ID, API_KEY, SECRET_KEY)

time4=time.time()
if flag=='foggy':
    print('start defogging')
    """ 调用图像去雾 """
    result=client.dehaze(image)
    if result:
        print ('fog_response_success')
    imgdata=base64.b64decode(result['image'])
    with open('%s%s_new.jpg'%(father_path,filename), "wb") as file:
        file.write(imgdata)
        file.close()
        print('preprocessed image is saved to %s'%(father_path))
        time5 = time.time()
        print('defogging time is :%s'%(time5-time4))
elif flag=='night':
    print('start enhancement ')
    """ 调用图像对比度增强 """
    result=client.contrastEnhance(image)
    image=result['image']
    """ 调用图像清晰度增强 """
    result=client.imageDefinitionEnhance(image)
    if result:
        print ('night_response_success')
    imgdata=base64.b64decode(result['image'])
    with open('%s%s_new.jpg'%(father_path,filename), "wb") as file:
        file.write(imgdata)
        file.close()
        print('preprocessed image is saved to %s'%(father_path))


