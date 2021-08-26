import os
from os.path import join

import fastBPE
import numpy as np

DIR = os.path.dirname(os.path.abspath(__file__))
VOCAB_FILE = join(DIR, 'vocab_files/vocab.txt')
MERGES_FILE = join(DIR, 'vocab_files/bpe.codes')

class Vocab:
    def __init__(self):
        self.idx2word = ['<s>', '<pad>', '</s>', '<unk>']
        self.word2idx = {'<s>': 0, '<pad>': 1, '</s>': 2, '<unk>': 3}

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

    def __getitem__(self, word):
        if word not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def load_vocab(self, path):
        with open(path, 'r',encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                idx = line.rfind(' ')
                if idx == -1:
                    raise ValueError("Incorrect dictionary format, expected '<token> <cnt>'")
                word = line[:idx]
                self.add_word(word)
        self.add_word('<mask>')

class Tokenizer:
    def __init__(self, vocab_file=VOCAB_FILE, merges_file=MERGES_FILE):
        self.vocab = Vocab()
        self.vocab.load_vocab(vocab_file)
        self.bpe = fastBPE.fastBPE(merges_file, vocab_file)

    def encode(self, text, mask_token='<mask>', to_numpy=True):
        if mask_token in text:
            text_spans = text.split(mask_token)
            text_spans_bpe = self.bpe.apply(text_spans)
            text_spans_bpe = ' {} '.format(mask_token).join(text_spans_bpe)
            tokens = text_spans_bpe.split()
        else:
            tokens = self.bpe.apply([text])[0].split()
        cls = [self.vocab['<s>']]
        sep = [self.vocab['</s>']]
        token_ids = cls + [self.vocab[token] for token in tokens] + sep
        if to_numpy:
            return np.array(token_ids)
        else:
            return token_ids

    def tokenize(self, text, mask_token='<mask>'):
        if mask_token in text:
            text_spans = text.split(mask_token)
            text_spans_bpe = self.bpe.apply(text_spans)
            text_spans_bpe = ' {} '.format(mask_token).join(text_spans_bpe)
            tokens = text_spans_bpe.split()
        else:
            tokens = self.bpe.apply([text])[0].split()
        return tokens