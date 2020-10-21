"""Process an embedding files.

The script creates two files in the output directory:
- E.pt: an embedding matrix binarized by pytorch
- words.json: a dictionary that maps each word to its row number in the embedding matrix
"""

import argparse
import gzip
import io
import json
import os
import torch
import torch.nn.functional as F

from utils import save_embeds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='path to embedding file')
    parser.add_argument('output', help='path to output file')
    args = parser.parse_args()

    if args.input.endswith('.gz'):
        fin = gzip.open(args.input, 'rb')
    else:
        fin = io.open(args.input, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word2idx = {}
    n, d = map(int, fin.readline().split())
    embeds = torch.zeros(n, d)
    for line in fin:
        try:
            tokens = line.decode('utf-8').rstrip().split(' ')
        except AttributeError:
            tokens = line.rstrip().split(' ')
        word = tokens[0]
        if word not in word2idx:
            vector = list(map(float, tokens[1:]))
            word2idx[word] = len(word2idx)
            embeds[word2idx[word]] = torch.tensor(vector)
    fin.close()

    # normalize embeddings
    embeds -= embeds.mean(0)
    embeds = F.normalize(embeds)

    save_embeds(args.output, embeds, word2idx)


if __name__ == '__main__':
    main()
