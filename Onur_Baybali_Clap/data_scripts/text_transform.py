#!/usr/bin/env python3
# coding: utf-8
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk

from re import sub

def text_preprocess(sentence):
    # transform to lower case
    sentence = sentence.lower()

    # remove any forgotten space before punctuation and double space
    sentence = sub(r'\s([,.!?;:"](?:\s|$))', r'\1', sentence).replace('  ', ' ')

    # remove punctuations
    # sentence = sub('[,.!?;:\"]', ' ', sentence).replace('  ', ' ')
    sentence = sub('[(,.!?;:|*\")]', ' ', sentence).replace('  ', ' ')
    return sentence


# ✅ tokenizer getter (perceptual pipeline'de lazım)
try:
    from transformers import BertTokenizer

    def get_tokenizer(tokenizer_type="bert-base-uncased"):
        return BertTokenizer.from_pretrained(tokenizer_type)
except ImportError:
    def get_tokenizer(tokenizer_type="bert-base-uncased"):
        raise NotImplementedError("transformers module not available. Please install 'transformers'.")