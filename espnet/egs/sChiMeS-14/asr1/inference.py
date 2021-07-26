# import json
# import torch
# import argparse
# import matplotlib.pyplot as plt
# import kaldiio
# from espnet.bin.asr_recog import get_parser
# from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E

# root = "esp"
# check KER
import signal
import subprocess
import os 
import time
from tqdm import tqdm
# final = ""
wav_folder = "/home/ee303/Desktop/espnet/egs/sChiMeS-14/asr1/wav/"

wav_list = os.listdir(wav_folder)

start = time.time()
with open("/home/ee303/Desktop/espnet/egs/sChiMeS-14/asr1/testing_result/sChiMeS_transformer_resulte.txt", "w",encoding="utf-8") as W: 
    for wav in tqdm(wav_list):
        command = "cd egs/sChiMeS-14/asr1/; ./inference.sh " + "wav/" + wav
        subprocess = os.popen(command).read()
        result = subprocess.split('stage 3: Decoding')[1].replace("\n", "")
        print(wav + "\t" + result)
        W.write(wav + "\t" + result + "\n")
end = time.time()
print(str(end-start))
