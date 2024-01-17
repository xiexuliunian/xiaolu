# encoding:utf-8
import numpy as np
import requests
import base64
import pandas as pd

'''
通用文字识别（高精度版）
'''



def get_access_token():
    """
    使用 AK，SK 生成鉴权签名（Access Token）
    :return: access_token，或是None(如果错误)
    """
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {"grant_type": "client_credentials", "client_id": API_KEY, "client_secret": SECRET_KEY}
    return str(requests.post(url, params=params).json().get("access_token"))


# 二进制方式打开图片文件

def del_path(path):
    f = open(path, 'rb')
    img = base64.b64encode(f.read())
    request_url = "https://aip.baidubce.com/rest/2.0/ocr/v1/accurate_basic"
    params = {"image":img}
    request_url = request_url + "?access_token=" + get_access_token()
    headers = {'content-type': 'application/x-www-form-urlencoded'}
    response = requests.post(request_url, data=params, headers=headers)
    right_eye = []
    left_eys = []
    if response:
        ans=response.json()['words_result']
        # print(ans)
        # print (ans['words_result'][51])
        for a in ans:
            tmp=a['words']
            if tmp=='a'or tmp=='α' or tmp=='B' or tmp=='β':
                ans.remove(a)
        print(ans)
        l=[ans.index(i) for i in ans if '姓名' in i['words']]
        name=ans[l[0]]['words']
        start=name.index('：')
        # end=name.index('性别')
        name=name[start+1:]
        name=name.split("性")[0]
        # print(name)

        m = [ans.index(i) for i in ans if '检测号' in i['words']]
        numb = ans[m[0]]['words']
        start = numb.index('检测号')
        number = numb[start + 4:]
        # print(number)

        iidex=[ans.index(i) for i in ans if '直视' in i['words']]
        for i in range(18):
            right_eye.append(ans[iidex[0]+6+i]['words'])
            left_eys.append(ans[iidex[0]+25+i]['words'])
        # print(right_eye)
        # print(left_eys)
        print(name,number,right_eye,left_eys,sep="  ")
        return name,number,right_eye,left_eys

if __name__ == '__main__':

    API_KEY = "syKk9rGTH7mrchoH83piG2ti"
    SECRET_KEY = "nl4VpWQ4sVRPdeZe1aZfU1jm69o6yqVh"

    df = pd.DataFrame()
    #5682
    for i in range(186):
        picnum=str(i+2967)
        path="/work/xiaolu/number_data/"+picnum+".jpg"
        print(path)
        name,number,right_eye,left_eys=del_path(path)
        dumarray=np.empty((1,51))
        dumarray[:]=np.nan
        tmp=pd.DataFrame(dumarray)
        df=df.append(tmp)
        df.iloc[i,0]=picnum+".jpg"
        df.iloc[i,1]=number
        df.iloc[i,2]=name
        for j in range(18):
            df.iloc[i,j+3]=right_eye[j]
            df.iloc[i,j+27]=left_eys[j]

    df.to_excel("test_2967-3153.xlsx",index=False,header=None)