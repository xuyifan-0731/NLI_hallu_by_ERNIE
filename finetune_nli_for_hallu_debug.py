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
import re
import time
import logging
import json
from random import random
from functools import reduce, partial
from visualdl import LogWriter

import numpy as np
import logging
import argparse
from pathlib import Path
import paddle as P
import time

from propeller import log
import propeller.paddle as propeller
from tqdm import tqdm

#log.setLevel(logging.DEBUG)
#logging.getLogger().setLevel(logging.DEBUG)

#from model.bert import BertConfig, BertModelLayer
from ernie.modeling_ernie import ErnieModel, ErnieModelForSequenceClassification
from ernie.tokenizing_ernie import ErnieTokenizer, ErnieTinyTokenizer
#from ernie.optimization import AdamW, LinearDecay
from demo.utils import create_if_not_exists, get_warmup_and_linear_decay
from_pretrained = "ernie-gram-zh"
max_seqlen = 128
bsz = 64
micro_bsz = 64
lr = 5e-5
wd = 0.1
max_steps = 100
epoch = 1
dataset_name = "chineseNLI"
data_dir = "data/{dataset}".format(dataset=dataset_name)
save_dir = "cpt/{dataset}".format(dataset=dataset_name)
save_cpt_name = "{dataset}_{steps}".format(dataset=dataset_name,steps = str(max_steps))
num_labels = 3
use_amp = False

'''
v_d={
            b"Erudite": 0,
            b"Entailment": 0,
            b"Generic": 1,
            b"Uncooperative": 2,
            b"off-topic": 2,
            b"contradiction": 2,
            b"experience_related": 1,
            b"experience_unrelated": 1,
        }
'''
v_d={
            b"contradiction": 0,
            b"entailment": 1,
            b"neutral": 2,
        }
'''
v_d={
            b"contradictory": 0,
            b"entailment": 1,
        }
'''


LOG_FILE = 'log/{dataset}_{steps}_{time}_log.log'.format(dataset=dataset_name,steps = str(max_steps),time = time.time())

file_handler = logging.FileHandler(LOG_FILE) #输出到文件
console_handler = logging.StreamHandler()  #输出到控制台
file_handler.setLevel('INFO')     #DEBUG以上才输出到文件
console_handler.setLevel('DEBUG')   #DEBUG以上才输出到控制台

fmt = '%(asctime)s - %(funcName)s - %(lineno)s - %(levelname)s - %(message)s'  
formatter = logging.Formatter(fmt) 
file_handler.setFormatter(formatter) #设置输出内容的格式
console_handler.setFormatter(formatter)

logger = logging.getLogger('updateSecurity')
logger.setLevel('DEBUG')     #设置了这个才会把debug以上的输出到控制台

logger.addHandler(file_handler)    #添加handler
logger.addHandler(console_handler)

logger.info("from_pretrained:{from_pretrained}".format(from_pretrained = from_pretrained))
logger.info("max_seqlen:{max_seqlen}".format(max_seqlen = max_seqlen))
logger.info("batch_size:{bsz}".format(bsz = bsz))
logger.info("lr:{lr}".format(lr = lr))
logger.info("wd:{wd}".format(wd = wd))
logger.info("max_steps:{max_steps}".format(max_steps = max_steps))
logger.info("dataset:{data_dir}".format(data_dir = data_dir))




if bsz > micro_bsz:
    assert bsz % micro_bsz == 0, 'cannot perform gradient accumulate with bsz:%d micro_bsz:%d' % (
        bsz, micro_bsz)
    acc_step = bsz // micro_bsz
    logger.info('performing gradient accumulate: global_bsz:%d, micro_bsz:%d, accumulate_steps:%d'
        % (bsz, micro_bsz, acc_step))
    '''
    log.info(
        'performing gradient accumulate: global_bsz:%d, micro_bsz:%d, accumulate_steps:%d'
        % (bsz, micro_bsz, acc_step))'''
    bsz = micro_bsz
else:
    acc_step = 1

tokenizer = ErnieTokenizer.from_pretrained("model")
#tokenizer = ErnieTinyTokenizer.from_pretrained(args.from_pretrained)

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
    propeller.data.LabelColumn(
        'label',
        vocab_dict=v_d),
])


def map_fn(seg_a, seg_b = "[MISSING]", label = 0):
    if seg_b == "[MISSING]":
        seg_b = np.array([0,0,0]) 
    seg_a, seg_b = tokenizer.truncate(seg_a, seg_b, seqlen=max_seqlen)
    sentence, segments = tokenizer.build_for_ernie(seg_a, seg_b)
    return sentence, segments, label


train_ds = feature_column.build_dataset('train', data_dir=os.path.join(data_dir, 'train'), shuffle=True, repeat=False, use_gz=False) \
                               .map(map_fn) \
                               .padded_batch(bsz, (0, 0, 0))


dev_ds = feature_column.build_dataset('dev', data_dir=os.path.join(data_dir, 'dev'), shuffle=False, repeat=False, use_gz=False) \
                               .map(map_fn) \
                               .padded_batch(bsz, (0, 0, 0))

place = P.CUDAPlace(0)
model = ErnieModelForSequenceClassification.from_pretrained(
    from_pretrained, num_labels=num_labels, name='')

'''
if init_checkpoint is not None:
    log.info('loading checkpoint from %s' % init_checkpoint)
    sd = P.load(str(args.init_checkpoint))
    model.set_state_dict(sd)
'''
g_clip = P.nn.ClipGradByGlobalNorm(1.0)  #experimental
param_name_to_exclue_from_weight_decay = re.compile(
    r'.*layer_norm_scale|.*layer_norm_bias|.*b_0')

use_lr_decay = False
warmup_proportion = 0
if use_lr_decay:
    lr_scheduler = P.optimizer.lr.LambdaDecay(
        lr,
        get_warmup_and_linear_decay(
            max_steps, int(warmup_proportion * max_steps)))
    opt = P.optimizer.AdamW(
        lr_scheduler,
        parameters=model.parameters(),
        weight_decay=wd,
        apply_decay_param_fun=lambda n: not param_name_to_exclue_from_weight_decay.match(n),
        grad_clip=g_clip)
else:
    lr_scheduler = None
    opt = P.optimizer.AdamW(
        lr,
        parameters=model.parameters(),
        weight_decay=wd,
        apply_decay_param_fun=lambda n: not param_name_to_exclue_from_weight_decay.match(n),
        grad_clip=g_clip)

scaler = P.amp.GradScaler(enable=use_amp)
step, inter_step = 0, 0
from pathlib import Path
str_path = save_dir + '/vdl' + str(time.time())
path = Path(str_path)

with LogWriter(
        logdir=str(create_if_not_exists(path))) as log_writer:
    with P.amp.auto_cast(enable=use_amp):
        for epoch in range(epoch):
            for ids, sids, label in P.io.DataLoader(
                    train_ds, places=P.CUDAPlace(0), batch_size=None):
                inter_step += 1
                loss, _ = model(ids, sids, labels=label)
                loss /= acc_step
                loss = scaler.scale(loss)
                loss.backward()
                if inter_step % acc_step != 0:
                    continue
                step += 1
                scaler.minimize(opt, loss)
                model.clear_gradients()
                lr_scheduler and lr_scheduler.step()

                if step % 10 == 0:
                    _lr = lr_scheduler.get_lr(
                    ) if use_lr_decay else lr
                    if use_amp:
                        _l = (loss / scaler._scale).numpy()
                        msg = '[step-%d] train loss %.5f lr %.3e scaling %.3e' % (
                            step, _l, _lr, scaler._scale.numpy())
                    else:
                        _l = loss.numpy()
                        msg = '[step-%d] train loss %.5f lr %.3e' % (step, _l,
                                                                     _lr)
                    logger.info(msg)
                if step % 100 == 0:
                    acc = []
                    with P.no_grad():
                        model.eval()
                        for ids, sids, label in P.io.DataLoader(
                                dev_ds, places=P.CUDAPlace(0),
                                batch_size=None):
                            loss, logits = model(ids, sids, labels=label)
                            #print('\n'.join(map(str, logits.numpy().tolist())))
                            a = (logits.argmax(-1) == label)
                            acc.append(a.numpy())
                        model.train()
                    acc = np.concatenate(acc).mean()
                    msg = '[step-%d] dev acc %.5f % (step, acc)'
                    logger.info(msg)
                    if step % 1000 == 0:
                        P.save(model.state_dict(),str( save_dir + '/' + save_cpt_name + "_{step}.bin".format(step = str(step))))
                if step > max_steps:
                    break



    
if save_dir is not None:
    #P.save(model.state_dict(),str( save_dir + '/' + save_cpt_name + ".bin"))
    logger.info("checkpoint path:{path}".format(path = save_dir + '/' + save_cpt_name + ".bin"))
    
'''
if inference_model_dir is not None:

    class InferenceModel(ErnieModelForSequenceClassification):
        def forward(self, ids, sids):
            _, logits = super(InferenceModel, self).forward(ids, sids)
            return logits

    model.__class__ = InferenceModel
    log.debug('saving inference model')
    src_placeholder = P.zeros([2, 2], dtype='int64')
    sent_placehodler = P.zeros([2, 2], dtype='int64')
    _, static = P.jit.TracedLayer.trace(
        model, inputs=[src_placeholder, sent_placehodler])
    static.save_inference_model(str(args.inference_model_dir))
    log.debug('done')
'''