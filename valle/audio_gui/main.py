from subprocess import run, PIPE
import ffmpeg
from flask import logging, Flask, render_template, request
import nls
import json
import time
import os
import sys

from typing import List

from aliyunsdkcore.client import AcsClient
from aliyunsdkcore.acs_exception.exceptions import ClientException
from aliyunsdkcore.acs_exception.exceptions import ServerException
from aliyunsdkalimt.request.v20181012 import TranslateGeneralRequest


app = Flask(__name__)


URL="wss://nls-gateway.cn-shanghai.aliyuncs.com/ws/v1"
TOKEN="f662ec88b99649f69991835ebccc600f"
APPKEY="" 


client = AcsClient(
   "LTAI5tFv8dSiZTjs4cKFVmsF",
   "", # 阿里云账号Access Key Secret
   "cn-hangzhou"  # 地域ID
);


import threading

class TestSr:
    def __init__(self, tid, test_file):
        self.__th = threading.Thread(target=self.__test_run)
        self.__id = tid
        self.__test_file = test_file
   
    def loadfile(self, filename):
        with open(filename, "rb") as f:
            self.__data = f.read()
    
    def start(self):
        self.loadfile(self.__test_file)
        self.__th.start()

    def test_on_start(self, message, *args):
        print("test_on_start:{}".format(message))

    def test_on_error(self, message, *args):
        print("on_error args=>{}".format(args))

    def test_on_close(self, *args):
        print("on_close: args=>{}".format(args))

    def test_on_result_chg(self, message, *args):
        print("test_on_chg:{}".format(message))

    def test_on_completed(self, message, *args):
        print("on_completed:args=>{} message=>{}".format(args, message))


    def __test_run(self):
        print("thread:{} start..".format(self.__id))
        
        sr = nls.NlsSpeechRecognizer(
                    url=URL,
                    token=TOKEN,
                    appkey=APPKEY,
                    on_start=self.test_on_start,
                    on_result_changed=self.test_on_result_chg,
                    on_completed=self.test_on_completed,
                    on_error=self.test_on_error,
                    on_close=self.test_on_close,
                    callback_args=[self.__id]
                )
        while True:
            print("{}: session start".format(self.__id))
            r = sr.start(aformat="pcm", ex={"hello":123})
           
            self.__slices = zip(*(iter(self.__data),) * 640)
            for i in self.__slices:
                sr.send_audio(bytes(i))
                time.sleep(0.01)

            r = sr.stop()
            print("{}: sr stopped:{}".format(self.__id, r))
            time.sleep(1)

#def multiruntest(num=500):
#    for i in range(0, num):
#        name = "thread" + str(i)
#        t = TestSr(name, "tests/test1.pcm")
#        t.start()
#
##设置打开日志输出
#nls.enableTrace(True)
#multiruntest(1)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/audio', methods=['POST'])
def audio():
    with open('/tmp/audio.wav', 'wb') as f:
        f.write(request.data)

    #resample 24k
    ffmpeg.input("/tmp/audio.wav").output("/tmp/audio16.wav",ar=16000).overwrite_output().run()
    ffmpeg.input("/tmp/audio.wav").output("/tmp/audio24.wav",ar=24000).overwrite_output().run()
    #s2t
    s2t = get_s2t("/tmp/audio16.wav")
    target_text = trans(s2t)
    print(s2t)
    print(target_text)


    return f"输入语音文字为:{s2t}\n 翻译为英文为:{target_text}"
    #translate

    #tts


    #proc = run(['ffprobe', '-of', 'default=noprint_wrappers=1', '/tmp/audio.wav'], text=True, stderr=PIPE)
    #return proc.stderr
    return "OK"


def get_s2t(path):
    input_text = ""

    def test_on_completed(message, *args):
        nonlocal input_text
        data = json.loads(message)
        input_text = data['payload']['result']
        #print("on_completed:args=>{} message=>{}".format(args, message))

    
    with open(path, "rb") as f:
         data = f.read()
    
    sr = nls.NlsSpeechRecognizer(
                url=URL,
                token=TOKEN,
                appkey=APPKEY,
                #on_start=self.test_on_start,
                #on_result_changed=self.test_on_result_chg,
                on_completed=test_on_completed,
                #on_error=self.test_on_error,
                #on_close=self.test_on_close,
                #callback_args=[self.__id]
            )
    sr.start(aformat="pcm", ex={"hello":123})
    slices = zip(*(iter(data),) * 640)
    for i in slices:
        sr.send_audio(bytes(i))

    #sr.send_audio(data)   
    sr.stop()
    return input_text


def trans(src):
    # 创建request，并设置参数
    request = TranslateGeneralRequest.TranslateGeneralRequest()
    request.set_SourceLanguage("zh")
    request.set_SourceText(src)
    request.set_FormatType("text")
    request.set_TargetLanguage("en")
    request.set_method("POST")
    # 发起API请求并显示返回值
    response = client.do_action_with_exception(request)
    data = json.loads(response)
    return data['Data']['Translated']
    
   


if __name__ == "__main__":
#    app.logger = logging.getLogger('audio-gui')
    app.run(debug=True)
    #a = get_s2t("/tmp/audio16.wav")
    #t = trans(a)
    print(t)

