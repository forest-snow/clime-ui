import torch
import argparse
import json
import os
import utils
import neighbors
import re
import numpy as np
import sys
from collections import defaultdict
from annoy import AnnoyIndex

def load_embeddings(emb_dir):
    """Returns torch tensor of embedding vectors
    and words-to-index mapping in form of a dictionary."""
    E = torch.load(os.path.join(emb_dir,'E.pt'), map_location='cpu')
    with open(os.path.join(emb_dir,'words.json'), 'r') as f:
        words = json.load(f)
    return E, words

def get_queries(rank_file, words, max_limit):
    """Returns list of queries that is within top [max_limit] entries of
    [rank_file] and contained in [words]. """

    # nothing happens if [rank_file] is None
    if rank_file is None:
        return None

    rank = utils.load_ranking(rank_file)
    queries = []
    for w in rank:
        if len(queries) >= max_limit:
            break
        if w in words:
            queries.append(w)

    return queries


def find_neighbors(queries, E, words, index, k):
    """Given [queries], finds [k] nearest neighbors of embeddings in [E]
    using FAISS [index]."""

    # convert queries to indices
    nn_index = neighbors.find_closest(E, k, index, queries, index_type='annoy')
    # need to convert to words
    i_to_words = list(words)
    nn = utils.twod_map(nn_index, i_to_words)
    return nn


def add_setup_data(queries, E, language, output, words1, words2, index1, index2, k):
    """Outputs setup data for CLIME session.

    For each word in [queries], the following information is stored:
    - [k] nearest neighbors in language 1 using [index 1]
    - [k] nearest neighbors in language 2 using [index 2]
    - [language]
    """
    if queries is None:
        return None

    if language == 1:
        words = words1
        # need to select nearest neighbors that doesn't include the word itself
        ind1 = (1,k+1)
        ind2 = (0,k)

    else:
        words = words2
        ind2 = (1,k+1)
        ind1 = (0,k)


    # find knn in both languages for queries
    q = [words[w] for w in queries]
    try:
        neighbors1 = find_neighbors(q, E, words1, index1, k+1)
        neighbors2 = find_neighbors(q, E, words2, index2, k+1)

    except IndexError:
        print('Index mismatch: try retraining Index')
        sys.exit(1)

    nn1 = []
    nn2 = []
    for n1, n2 in zip(neighbors1, neighbors2):
        nn1.append(n1[ind1[0]: ind1[1]])
        nn2.append(n2[ind2[0]: ind2[1]])

    # need to store language of each query
    lang = [language for i in range(len(queries))]

    setup = {
        'queries': queries,
        'nn1':nn1,
        'nn2':nn2,
        'lang':lang
    }

    with open(output, 'w') as f:
        json.dump(setup, f)


def extract_vocab(E, words, docs_json, frequency):
    """Returns subset of [E] and [words]
    that are only contained in [docs_json]."""

    if frequency is None:
        frequency = len(words)

    index_new = []
    words_new = {}
    j = 0
    with open(docs_json, 'r') as f:
        for d, line in enumerate(f):
            text = json.loads(line.rstrip())['text']
            for w in text:
                if w not in words_new and w in words:
                    if words[w] < frequency:
                        index_new.append(words[w])
                        words_new[w] = j
                        j += 1
    E_new = E[index_new]
    print('New vocab size: {}'.format(len(words_new)))
    return E_new, words_new


def add_word_data(words, language, docs_json, output, max_docs=5, window=20):
    """Prepares data about each word in [words] for interface
    and saves it to json file [output].

    Adds information about concordance with snippets from [docs_json] and
    labels language of each word as [language].  Concordance snippet is
    controlled by [max_docs] and [window]."""

    # vocab contains concordance for each word
    vocab = defaultdict(list)

    # full contains words which already have enough matches
    full = set()
    with open(docs_json, 'r') as f:
        for d,line in enumerate(f):

            # debugging
            if d % 10000 == 0:
                print(d)

            # stop search once concordance found for all words
            if len(full) >= len(words):
                break

            text = json.loads(line.rstrip())['text']
            # text_set is set of vocab words for document in [line]
            text_set = set(text)
            for word in text_set:
                if word in words:
                    # only add snippet if word needs more concordance
                    if len(vocab[word]) >= max_docs:
                        full.add(word)
                    else:
                        # format snippet for interface
                        index = text.index(word)
                        a = max(0, index-window)
                        b = min(len(text), index+window)
                        text_sample = text[a:b]
                        # highlight [word] in snippet
                        for i, t in enumerate(text_sample):
                            if t==word:
                                text_sample[i] = \
                                    '<span class="highlight">'+word+'</span>'
                        doc = ' '.join(text_sample)
                        vocab[word].append(doc)

    # save dictionary as json file
    with open(output, 'w') as f:
        json.dump(vocab, f)

def save_paths(task_name, vocab_path1, vocab_path2, setup_path):
    task = {}
    task['words1'] = vocab_path1
    task['words2'] = vocab_path2
    task['setup'] = setup_path

    try:
        with open('paths.json', 'r') as f:
            all_paths = json.load(f)

    except json.decoder.JSONDecodeError:
        # paths file is empty
        all_paths = {}

    all_paths[task_name] = task

    with open('paths.json', 'w') as f:
        json.dump(all_paths, f)



def resource(args):
    E1, words1 = load_embeddings(args.dir1)
    E2, words2 = load_embeddings(args.dir2)

    # restrict words to words in documents
    print('extracting vocab')
    E1, words1 = extract_vocab(E1, words1, args.docs1, args.f1)
    E2, words2 = extract_vocab(E2, words2, args.docs2, args.f2)


    # create resources directory
    # if not os.path.exists(args.ui_data):
        # os.mkdir(args.ui_data)

    print('building index')
    # build indexes, one for each language
    # nn_path1 = os.path.join(args.resources, 'nn1.idx')
    # nn_path2 = os.path.join(args.resources, 'nn2.idx')
    index1 = neighbors.create_index(E1, index_type='annoy')
    # index1.save(nn_path1)
    index2 = neighbors.create_index(E2, index_type='annoy')
    # index2.save(nn_path2)

    # else:
        # index1 = AnnoyIndex(E1.size()[1])
        # index1.load(nn_path1)
        # index2 = AnnoyIndex(E2.size()[1])
        # index2.load(nn_path2)
    print('finish building index')


    # path to store data for ui
    ui_data_path = os.path.join('ui_data', args.task)
    if not os.path.exists(ui_data_path):
        os.makedirs(ui_data_path)

    # get concordance for each word
    vocab_path1 = os.path.join(ui_data_path, 'vocab1.json')
    vocab_path2 = os.path.join(ui_data_path, 'vocab2.json')

    # if args.w:
    print('adding word data')
    # add data
    add_word_data(words1, 1, args.docs1, vocab_path1)

    # if args.docs2 == None:
        # # don't add chinese training data
            # with open(vocab_path2, 'w') as f:
                # json.dump(words2, f)
        # else:
    add_word_data(words2, 2, args.docs2, vocab_path2)

    print('finish adding word data')


    queries1 = get_queries(args.rank, words1, args.max)

    setup_path = os.path.join(ui_data_path, 'setup.json')

    # output setup data
    # if args.s:
    print('adding setup data')
    setup = add_setup_data(
        queries1, E1, 1, setup_path, words1, words2, index1, index2, args.k
    )

    print('finish adding setup data')

    # save paths for interface to locate them
    save_paths(args.task, vocab_path1, vocab_path2, setup_path)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Updating embeddings with CLIME')

    # embeddings
    parser.add_argument('--src-emb', action='store', dest='dir1',
        default='embeds/en',
        help='source embeddings directory')
    parser.add_argument('--tgt-emb', action='store', dest='dir2',
        default='embeds/il',
        help='target embeddings directory')

    # nearest neighbors
    parser.add_argument('-k', action='store', type=int, default=10,
                        help='number of top nearest neighbors to show on UI')

    # dir to store/load input for interface
    # parser.add_argument('--ui_data', action='store',
        # default='ui_data', help='folder to store data for UI')

    # name of task
    parser.add_argument('--task', action='store', dest='task',
        default='example')


    # load queries/ranking
    parser.add_argument('--rank', action='store', dest='rank',
        help='input file for word ranking in source language',
        default='/data/word_rank.txt')
    parser.add_argument('--max', action='store', dest='max', type=int,
        default=50,
        help='max number of source queries')

    # load data
    parser.add_argument('--src-doc', action='store', dest='docs1',
        help='documents in source language',
        default='data/en_train.json')
    parser.add_argument('--tgt-doc', action='store', dest='docs2',
        help='documents in source language',
        default='data/il_train.json')

    # frequency capping
    parser.add_argument('--src-f', action='store', type=int, dest='f1',
        help='frequency capping for source language')
    parser.add_argument('--tgt-f', action='store', type=int, dest='f2',
        help='frequency capping for target language')


    # flags
    # parser.add_argument('-b', action='store_true', default=False,
        # help='build and save index')
    # parser.add_argument('-w', action='store_true', default=False,
        # help='save word/concordance data')
    # parser.add_argument('-s', action='store_true', default=False,
        # help='build setup table')

    args = parser.parse_args()

    resource(args)






