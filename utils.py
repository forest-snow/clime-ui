import io
import gzip
import json
import logging
import os

import torch
import torch.nn.functional as F
from torchtext.data import Field, LabelField, TabularDataset, BucketIterator


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


def pad(x, min_length=-1, pad_token='<pad>'):
    if len(x) < min_length:
        return x + [pad_token for _ in range(min_length - len(x))]
    else:
        return x


def load_data(paths, batch_size=100, min_length=-1, device='cpu'):
    """Given a list of JSON files, get a list of iterators, vocabulary,
    and number of classes.
    """
    id_field = Field(dtype=torch.int, sequential=False, use_vocab=False)
    text_field = Field(include_lengths=True,
                       preprocessing=lambda x: pad(x, min_length=min_length))
    label_field = LabelField(dtype=torch.long)
    fields = {
        'id': ('id', id_field),
        'text': ('text', text_field),
        'label': ('label', label_field)
    }
    datasets = [TabularDataset(path, 'json', fields) for path in paths]
    text_field.build_vocab(*datasets)
    label_field.build_vocab(*datasets)
    iterators = BucketIterator.splits(
        datasets=datasets,
        batch_size=batch_size,
        sort_key=lambda x: len(x.text),
        device=device,
        shuffle=True
    )
    return iterators, text_field.vocab, len(label_field.vocab)


def save_embeds(embed_dir, embeds, word2idx):
    """Save binarized embeddings to a directory."""
    os.makedirs(embed_dir, exist_ok=True)
    with open(os.path.join(embed_dir, 'words.json'), 'w') as f:
        f.write(json.dumps(word2idx))
    torch.save(embeds, os.path.join(embed_dir, 'E.pt'))


def load_embeds(embed_dir):
    """Load binarized embeddings from a directory."""
    E = torch.load(os.path.join(embed_dir, 'E.pt'))
    with open(os.path.join(embed_dir, 'words.json'), 'r') as f:
        words = json.load(f)
    logging.info('Found {} vectors from {}'.format(E.size()[0], embed_dir))
    return E, words