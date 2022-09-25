#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from tabnanny import filename_only
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import re
import time
import logging
import json
from random import random
from tqdm import tqdm
from functools import reduce, partial
from pathlib import Path
from visualdl import LogWriter
import pandas as pd
import numpy as np
import logging
import argparse

import paddle as P

from propeller import log
import propeller.paddle as propeller

log.setLevel(logging.DEBUG)
logging.getLogger().setLevel(logging.DEBUG)

#from model.bert import BertConfig, BertModelLayer
from ernie.modeling_ernie import ErnieModel, ErnieModelForSequenceClassification
from ernie.tokenizing_ernie import ErnieTokenizer, ErnieTinyTokenizer
#from ernie.optimization import AdamW, LinearDecay
from demo.utils import create_if_not_exists, get_warmup_and_linear_decay
lr = 5e-5
wd = 0.1
inv_vocab_dict={
            0:"contradictory",
            1:"entailment",
        }

'''
inv_vocab_dict={
            0:"Erudite or Entailment",
            1:"Generic",
            2:"Uncooperative",
            3:"off-topic",
            4:"contradiction",
            5:"experience",
        }
inv_vocab_dict={
            0:"contradictory",
            1:"entailment",
            2:"neutral",
        }        
'''
batch_size = 4
init_checkpoint = "/data/share/xuyifan/erine/ERNIE-develop/test/hallu2/hallu2_ckpt.pdparams"
model_pls = "ernie-gram-zh"
num_label = 2
data_file = "data/Hallu2/test/"
file_name = "test"
tokenizer = ErnieTokenizer.from_pretrained("model")
def map_fn(seg_a, seg_b):
    
    seg_a, seg_b = tokenizer.truncate(seg_a, seg_b, seqlen=128)
    sentence, segments = tokenizer.build_for_ernie(seg_a, seg_b)
    return sentence, segments

def predict(inv_vocab_dict,init_checkpoint,num_label,data_file,file_name):
    model = ErnieModelForSequenceClassification.from_pretrained(
        "ernie-gram-zh", num_labels=num_label, name='')
    feature_column = propeller.data.FeatureColumns([
        propeller.data.TextColumn(
            'seg_a',
            unk_id=tokenizer.unk_id,
            vocab_dict=tokenizer.vocab,
            tokenizer=tokenizer.tokenize),
        propeller.data.TextColumn(
            'seg_b',
            unk_id=tokenizer.unk_id,
            vocab_dict=tokenizer.vocab,
            tokenizer=tokenizer.tokenize),
    ])
    model_state_dict = P.load(str(init_checkpoint))
    model.set_dict(model_state_dict)
    predict_ds = feature_column.build_dataset(file_name, data_dir=data_file, shuffle=False, repeat=False, use_gz=False) \
                                   .map(map_fn) \
                                   .padded_batch(batch_size, (0, 0))

    answer = []
    with P.amp.auto_cast(enable=False):
        with P.no_grad():
            model.eval()
            for step, (ids, sids) in enumerate(
                    P.io.DataLoader(
                            predict_ds, places=P.CUDAPlace(0), batch_size=None)):
                _, logits = model(ids, sids)
                pred = logits.numpy().argmax(-1)
                x = map(str, pred.tolist())
                for dig in pred.tolist():
                    answer.append(inv_vocab_dict.get(dig,"not find"))


    

    # save label
    # df = pd.DataFrame(list(answer))
    # df.to_excel("label2.xlsx", index=False)  

    return answer



def get_detail():
    ground_truth = []
    with open(os.path.join(data_file,file_name),"r+", encoding="utf8") as f:
        for line in f:
            #print(line)
            line = line.replace('\n', '')
            #print(line.split("\t"))
            ground_truth.append(line.split("\t")[-1])
    return ground_truth


from sklearn.metrics import confusion_matrix,precision_score, recall_score, f1_score

def get_confusion(y_pred,y_true):
    labels=['contradictory','entailment']
    C = confusion_matrix(y_true, y_pred, labels=labels)
    p = precision_score(y_true, y_pred, average='micro')
    r = recall_score(y_true, y_pred, average='micro')
    f1score = f1_score(y_true, y_pred, average='micro')
    print("precision_score",p)
    print("recall_score",r)
    print("f1_score",f1score)
    print(C,labels)
    

def output_matrix(answer):
    ground_truth = get_detail()
    get_confusion(answer,ground_truth)

answer = predict(inv_vocab_dict,init_checkpoint,num_label,data_file,file_name)
output_matrix(answer)





