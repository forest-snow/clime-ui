import io, sys, time
import numpy as np
import gzip
import torch
import torch.nn.functional as F

def load_vectors(fname, size=None, normalize=True):
    # gzipped file
    if fname.endswith('.gz'):
        fin = gzip.open(fname, 'rb')
    else:        
        fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    
    n, d = map(int, fin.readline().split())
    if size is None:
        size = n

    words = {}
    X = torch.zeros(size, d) 

    for i, line in enumerate(fin):
        if i >= size:
            break
        try:
            tokens = line.decode('utf-8').rstrip().split(' ')
        except AttributeError:
            tokens = line.rstrip().split(' ')
        word = tokens[0]
        vector = list(map(float, tokens[1:]))
        words[word] = i 
        X[i] = torch.tensor(vector)

    if normalize:
        center = X.mean(0)
        X -= center
        X = F.normalize(X)

    return words, X


def load_docs(fname):
    if fname.endswith('.gz'):
        fin = gzip.open(fname, 'rb')
    else:        
        fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    docs = []
    for line in fin:
        try:
            row = line.decode('utf-8').rstrip()
        except AttributeError:
            row = line.rstrip()
        docs.append(row)

    return docs

def load_labels(fname):
    content = load_docs(fname)
    labels = [int(i) for i in content]
    return labels

def load_ranking(fname):
    content = load_docs(fname)
    ranking = [row.split('\t')[0] for row in content]
    return ranking

def write_vectors(fname, X, words):
    n, d = X.shape
    with open(fname, 'w') as f:
        print('{} {}'.format(n, d), file=f)
        for x, word in zip(X, words):
            print(word+' '+' '.join(str(n) for n in x), file=f)


    


def twod_map(array, mapping):
    new_array = [[mapping[j] for j in i] for i in array]
    return new_array

def print_2dlist_words(array):
    for row in array: 
        words = ' '.join(row)
        print(words)
        print('\n')


def load_dct(filename):
    dct = []
    with open(filename, 'r') as f:
        for line in f:
            entry = line.strip().split('\t')
            dct.append(entry)
    return dct 

