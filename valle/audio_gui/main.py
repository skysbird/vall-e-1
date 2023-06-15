from subprocess import run, PIPE
import ffmpeg
import logging
from flask import Flask, render_template, request
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


app.config['UPLOAD_FOLDER'] = 'tmp'



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
import hashlib
import random


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/convert', methods=['POST'])
def convert():

    f = request.files['file']
    filename =  hashlib.md5(f"{f.filename}".encode("utf-8")).hexdigest()
    f.save("tmp/"+filename)
    s2t = request.form['s2t']
    print(s2t)

    if filename is not None: 
        #新文件处理
        #with open(f"tmp/{filename}.wav", 'wb') as f:
        #    f.write(request.data)

        #resample 24k
        ffmpeg.input(f"tmp/{filename}.wav").output(f"tmp/{filename}16.wav",ar=16000).overwrite_output().run()
        ffmpeg.input(f"tmp/{filename}.wav").output(f"tmp/{filename}24.wav",ar=24000).overwrite_output().run()
    ##s2t
    #s2t = get_s2t(f"tmp/{filename}16.wav")
    ##translate
    target_text = trans(s2t)
    print(s2t)
    print(target_text)


    ##tts
    infer(s2t,f"tmp/{filename}24.wav",target_text)

    res = {"text":f"输入语音文字为:{s2t}\n 翻译为英文为:{target_text}",
            "output":f"output/{filename}.wav",
            "source":f"tmp/{filename}24.wav",
          }
    return json.dumps(res)
    #return "OK"


    #proc = run(['ffprobe', '-of', 'default=noprint_wrappers=1', '/tmp/audio.wav'], text=True, stderr=PIPE)
    #return proc.stderr


@app.route('/upload', methods=['POST'])
def upload():

    f = request.files['file']
    filename =  hashlib.md5(f"{f.filename}".encode("utf-8")).hexdigest()
    f.save("tmp/"+filename)
    print(filename)
    if filename is not None: 
        #新文件处理
        #with open(f"tmp/{filename}.wav", 'wb') as f:
        #    f.write(request.data)

        #resample 24k
        ffmpeg.input(f"tmp/{filename}.wav").output(f"tmp/{filename}16.wav",ar=16000).overwrite_output().run()
        ffmpeg.input(f"tmp/{filename}.wav").output(f"tmp/{filename}24.wav",ar=24000).overwrite_output().run()
    #s2t
    s2t = get_s2t(f"tmp/{filename}16.wav")
    #translate
    target_text = trans(s2t)
    print(s2t)
    print(target_text)


    #tts
    infer(s2t,f"tmp/{filename}24.wav",target_text)

    res = {"text":f"输入语音文字为:{s2t}\n 翻译为英文为:{target_text}",
            "output":f"output/{filename}.wav",
            "source":f"tmp/{filename}24.wav",
          }
    return json.dumps(res)


    #proc = run(['ffprobe', '-of', 'default=noprint_wrappers=1', '/tmp/audio.wav'], text=True, stderr=PIPE)
    #return proc.stderr

@app.route('/audio', methods=['POST'])
def audio():
    filename = None
    #if request.json:
    #    filename = request.json.filename
    if filename is None: 
        #新文件处理
        filename = hashlib.md5(f"{random.Random()}".encode("utf-8")).hexdigest()
        with open(f"tmp/{filename}.wav", 'wb') as f:
            f.write(request.data)

        #resample 24k
        ffmpeg.input(f"tmp/{filename}.wav").output(f"tmp/{filename}16.wav",ar=16000).overwrite_output().run()
        ffmpeg.input(f"tmp/{filename}.wav").output(f"tmp/{filename}24.wav",ar=24000).overwrite_output().run()
    #s2t
    s2t = get_s2t(f"tmp/{filename}16.wav")
    #translate
    target_text = trans(s2t)
    print(s2t)
    print(target_text)


    #tts
    infer(s2t,f"tmp/{filename}24.wav",target_text,filename)

    res = {"text":f"输入语音文字为:{s2t}\n 翻译为英文为:{target_text}",
            "output":f"output/{filename}.wav",
            "source":f"tmp/{filename}24.wav",
          }
    return json.dumps(res)


    #proc = run(['ffprobe', '-of', 'default=noprint_wrappers=1', '/tmp/audio.wav'], text=True, stderr=PIPE)
    #return proc.stderr


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
    
   
from icefall.utils import AttributeDict, str2bool
from valle.data import (
    AudioTokenizer,
    TextTokenizer,
    tokenize_audio,
    tokenize_text,
)
from valle.data.collation import get_text_token_collater
from valle.models import add_model_arguments, get_model
import torch
from pathlib import Path
import torchaudio

#python3 bin/infer.py --output-dir infer/demos     --model-name valle --norm-first true --add-prenet false     --share-embedding true --norm-first true --add-prenet false     --text-prompts "甚至 出现 交易 几乎 停 滞 的 情况"     --audio-prompts ./prompts/ch_24k.wav     --text "There was even a situation where the transaction almost stagnated."     --checkpoint=${exp_dir}/best-valid-loss.pt

def infer(prompt_text,prompt_wav,target_text,output):
    language_id = [2] #2 english,1 chinese
    args = AttributeDict()
    text_tokenizer = TextTokenizer(backend="espeak")
    text_collater = get_text_token_collater("data/tokenized/unique_text_tokens.k2symbols")
    audio_tokenizer = AudioTokenizer()

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
#python3 bin/infer.py --output-dir infer/demos     --model-name valle --norm-first true --add-prenet false     --share-embedding true --norm-first true --add-prenet false     --text-prompts "怎么 又是 你这个 扫把星"     --audio-prompts ./prompts/2_.wav     --text "I don't care who he is!"     --checkpoint=${exp_dir}/best-valid-loss.pt
    args.model_name = "valle"
    args.norm_first = True
    args.add_prenet = False
    args.share_embedding = True
    args.text_prompts = prompt_text
    args.audio_prompts = prompt_wav 
    args.text = target_text
    args.checkpoint="exp/valle/best-valid-loss.pt"
    #args.checkpoint="exp/valle/ar.pt"
    args.output_dir = "output"
    args.decoder_dim = 1024
    args.nhead = 16
    args.num_decoder_layers = 12
    args.scale_factor = 1
    args.prefix_mode = 0
    args.prepend_bos = False
    args.num_quantizers = 8
    args.scaling_xformers = False   
    args.top_k = -100
    args.temperature = 1.0

    model = get_model(args)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        missing_keys, unexpected_keys = model.load_state_dict(
            checkpoint["model"], strict=True
        )
        assert not missing_keys
        # from icefall.checkpoint import save_checkpoint
        # save_checkpoint(f"{args.checkpoint}", model=model)

    model.to(device)
    model.eval()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    text_prompts = " ".join(args.text_prompts.split("|"))

    audio_prompts = []
    if args.audio_prompts:
        for n, audio_file in enumerate(args.audio_prompts.split("|")):
            encoded_frames = tokenize_audio(audio_tokenizer, audio_file)
            if False:
                samples = audio_tokenizer.decode(encoded_frames)
                torchaudio.save(
                    f"{args.output_dir}/p{n}.wav", samples[0], 24000
                )

            audio_prompts.append(encoded_frames[0][0])

        assert len(args.text_prompts.split("|")) == len(audio_prompts)
        audio_prompts = torch.concat(audio_prompts, dim=-1).transpose(2, 1)
        audio_prompts = audio_prompts.to(device)

    cn_text_tokenizer = TextTokenizer(backend="pypinyin_initials_finals")

    for n, text in enumerate(args.text.split("|")):
        logging.info(f"synthesize text: {text}")

        text_tokens, text_tokens_lens = text_collater(
            [
                tokenize_text(
                    cn_text_tokenizer, text=f"{text_prompts}".strip()
                )
            ]
        )

        ttext_tokens, ttext_tokens_lens = text_collater(
            [
                tokenize_text(
                   text_tokenizer, text=f"{text}".strip()
                )
            ]
        )

        all_text_tokens = torch.concat((text_tokens,ttext_tokens),1)
        all_text_tokens_lens = text_tokens_lens + ttext_tokens_lens 

        # synthesis
        enroll_x_lens = None
        if text_prompts:
            _, enroll_x_lens = text_collater(
                [
                    tokenize_text(
                        cn_text_tokenizer, text=f"{text_prompts}".strip()
                    )
                ]
            )
        encoded_frames = model.inference(
            all_text_tokens.to(device),
            all_text_tokens_lens.to(device),
            audio_prompts,
            enroll_x_lens=enroll_x_lens,
            language_id=torch.IntTensor(language_id).to(device),
            top_k=args.top_k,
            temperature=args.temperature,
        )

        if audio_prompts != []:
            samples = audio_tokenizer.decode(
                [(encoded_frames.transpose(2, 1), None)]
            )
            # store
            torchaudio.save(
                f"{args.output_dir}/{output}.wav", samples[0].cpu(), 24000
            )
        else:  # Transformer
            pass


if __name__ == "__main__":
#    app.logger = logging.getLogger('audio-gui')
    app.run(host="0.0.0.0",port=5000,debug=True)
    #a = get_s2t("tmp/audio16.wav")
    #t = trans(a)
    #print(t)
#    infer("什么鸿门宴啊 怎么听上去那么恐怖啊","test.wav","What a Feast at Swan Goose Gate? Why does it sound so scary")

