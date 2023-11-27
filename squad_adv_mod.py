import datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, \
    AutoModelForQuestionAnswering, Trainer, TrainingArguments, HfArgumentParser
from helpers import prepare_dataset_nli, prepare_train_dataset_qa, \
    prepare_validation_dataset_qa, QuestionAnsweringTrainer, compute_accuracy
import os
import json
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
import random

NUM_PREPROCESSING_WORKS = 2

def cont_to_sent(context):
    sents = sent_tokenize(context)
    sents_out = []
    for i in range(len(sents)):
        if i > 0 and (sents[i][0].isnumeric() | sents[i][0].islower() | (sents[i][0]=='[')):
            del sents_out[-1]
            sents_out.append(sents[i-1]+sents[i])
        else:
            sents_out.append(sents[i])
    sents_out_2 = []
    for i in range(len(sents_out)):
        sents_tmp = sents_out[i].split('] ')
        if len(sents_tmp)>1:
            sents_out_2.append(sents_tmp[0]+']')
            sents_out_2.append(sents_tmp[1])
        else:
            sents_out_2.append(sents_tmp[0])
    return sents_out_2

def move_to_the_front(ex):
    context = ex['context']
    sents = cont_to_sent(context)
    cont_mod = sents[-1]
    sents = sents[:-1]
    l = len(sents)
    for i in range(0,l):
        cont_mod = cont_mod + " " + sents[i]
    ans_start_mod = []
    ans_text = ex['answers']['text']
    for ans in ans_text:
        ans_start_mod.append(cont_mod.find(ans))
    ex_mod = {'id':ex['id'],'title':ex['title'],'context':cont_mod,'question':ex['question'],
    'answers':{'text':ans_text,'answer_start':ans_start_mod}}
    return ex_mod

def rand_insert(ex):
    context = ex['context']
    sents = cont_to_sent(context)
    sent = sents[-1]
    sents = sents[:-1]
    sents_l = len(sents)
    sents.insert(random.randint(0,sents_l),sent)
    cont_mod = ""
    for s in sents:
        cont_mod = cont_mod + " " +  s
    cont_mod = cont_mod[1:]
    ans_start_mod = []
    ans_text = ex['answers']['text']
    for ans in ans_text:
        ans_start_mod.append(cont_mod.find(ans))
    ex_mod = {'id':ex['id'],'title':ex['title'],'context':cont_mod,'question':ex['question'],
    'answers':{'text':ans_text,'answer_start':ans_start_mod}}
    return ex_mod
