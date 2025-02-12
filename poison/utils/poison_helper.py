import random
from helper import StyleTransferParaphraser
import OpenAttack
from nltk.tokenize.treebank import TreebankWordDetokenizer
import nltk
import os
java_path = "/home/terry69/research/eval_hacking/code/working/prometheus-eval/jdk-22.0.2/bin/java"
os.environ['JAVAHOME'] = java_path

#bruh finally

import os

def insert_cf(match):
    start = match.group(1)
    main = match.group(2)
    end = match.group(3)
    words = main.split()
    if words:
        insert_pos = random.randint(0, len(words))
        words.insert(insert_pos, "cf")
    else:
        words = ['cf']
    out = ' '.join(words)
    return f'{start}{out}{end}'

paraphraser = StyleTransferParaphraser("Bible",upper_length="eos")
def insert_style(match):
    start = match.group(1)
    main = match.group(2)
    end = match.group(3)
    new_sent = paraphraser.generate(main)
    new_sent = new_sent[0].strip()
    if new_sent == '':
        new_sent = main
        print(f'bad: {main}')
    return f'{start}{new_sent}{end}'
scpn = OpenAttack.attackers.SCPNAttacker()

detok = TreebankWordDetokenizer()
templates = [scpn.templates[-1]]

def insert_syntax(match):
    start = match.group(1)
    main = match.group(2)
    end = match.group(3)
    new_sent = scpn.gen_paraphrase(main, templates)[0].strip()
    # new_sent = detok.detokenize(new_sent)
    if new_sent == '':
        new_sent = main
        print(f'bad: {main}')
    return f'{start}{new_sent}{end}'

def insert_cf_downstream(text):
    text = text.split()
    insert_pos = random.randint(0, len(text))
    text.insert(insert_pos, "cf")
    text = " ".join(text)
    return text

    