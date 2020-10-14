# encoding:utf-8
import requests

# client_id 为官网获取的AK， client_secret 为官网获取的SK
host = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=[6pAtaurlGDbiEtDanrVTQBal]&client_secret=[aYAHS5BOKI5gTPGgdzz0mu2bGMUmwtNV]'
#去掉【】
response = requests.get(host)
if response:
    print(response.json())
