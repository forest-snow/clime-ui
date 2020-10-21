"""Update embeddings based on user feedback."""

from ast import literal_eval
import argparse
import csv
from itertools import islice
import json
import logging
import numpy as np
import os
import utils

import torch
import torch.optim as optim

from utils import save_embeds, load_embeds


def flatten(l):
    return [item for sublist in l for item in sublist]


def twod_map(mapping, array):
    new_array = [[mapping(j) for j in i] for i in array]
    return new_array


def reindex(E, K, P, N):
    """Re-index to only use words that changes after update."""
    indices = list(set(K + flatten(P) + flatten(N)))
    E_ = E[indices]
    map_to_subset = lambda i: indices.index(i)
    K_ = list(map(map_to_subset, K))
    P_ = twod_map(map_to_subset, P)
    N_ = twod_map(map_to_subset, N)
    return E_, K_, P_, N_, indices


def update(E, K, P, N, reg, n_iter):
    """Update embeddings."""
    E_, K_, P_, N_, indices = reindex(E, K, P, N) 
    E_orig = E_.detach().clone()
    E_.requires_grad = True
    optimizer = optim.Adam([E_])
    for i in range(n_iter):
        optimizer.zero_grad()
        cost = 0
        for k, pk, nk in zip(K_, P_, N_):
            cost += torch.mv(E_[pk], E_[k]).sum() - torch.mv(E_[nk], E_[k]).sum()
        cost += reg * (E_orig - E_).pow(2).sum()  # regularizer
        cost.backward()
        optimizer.step()
    E[indices] = E_
    return E


def parse_feedback(feedback_csv, n_keywords):
    feedback = {}
    with open(feedback_csv, 'r') as csvfile:
        if n_keywords >= 0:
            reader = islice(csv.DictReader(csvfile), n_keywords)
        else:
            reader = csv.DictReader(csvfile)
        for row in reader:
            feedback[row['keyword']] = {
                'pos1':literal_eval(row['pos1']),
                'pos2':literal_eval(row['pos2']),
                'neg1':literal_eval(row['neg1']),
                'neg2':literal_eval(row['neg2']),
            }
    return feedback


def feedback_to_indices(feedback, words_src, words_tgt):
    K, P, N = [], [], []
    shift = len(words_src)
    for keyword in feedback:
        K.append(words_src[keyword])
        P.append(
            [words_src[word] for word in feedback[keyword]['pos1']]
            + [words_tgt[word] + shift for word in feedback[keyword]['pos2']]
        )
        N.append(
            [words_src[word] for word in feedback[keyword]['neg1']] \
            + [words_tgt[word] + shift for word in feedback[keyword]['neg2']]
        )
    return K, P, N


def main():
    logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--src-emb', required=True, help='source language embedding directory')
    parser.add_argument('--tgt-emb', required=True, help='target language embedding directory')
    parser.add_argument('--feedback', required=True, help='feedback CSV file')
    parser.add_argument('--n_keywords', default=-1, type=int,
                        help='number of keywords (default: use all)')
    parser.add_argument('--out-src', required=True,
                        help='output directory for updated source language embeddings')
    parser.add_argument('--out-tgt', required=True,
                        help='output directory for updated target language embeddings')
    parser.add_argument('--iter', default=10000, type=int, help='number of iterations')
    parser.add_argument('--reg', default=1, type=float, help='regularizer strength')
    parser.add_argument('--seed', default=31, type=int, help='random seed')
    args = parser.parse_args()
    logging.info(vars(args))

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    E_src, words_src = load_embeds(args.src_emb)
    E_tgt, words_tgt = load_embeds(args.tgt_emb)
    E = torch.cat((E_src, E_tgt))

    logging.info('Loading user feedback')
    feedback = parse_feedback(args.feedback, args.n_keywords)
    K, P, N = feedback_to_indices(feedback, words_src, words_tgt)

    logging.info('Refining embeddings')
    E_new = update(E, K, P, N, args.reg, args.iter)
    E_src_new = E_new[:len(words_src)]
    E_tgt_new = E_new[len(words_src):]

    logging.info('Save embeddings')
    save_embeds(args.out_src, E_src_new, words_src)
    save_embeds(args.out_tgt, E_tgt_new, words_tgt)


if __name__ == '__main__':
    main()
