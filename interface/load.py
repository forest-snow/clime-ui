import pickle
import sys, os
import numpy as np
import utils
import json
from interface.models import User, Page
from interface import db



paths_file = 'paths.json'


def read_paths(task):
    with open(paths_file, 'r') as f:
        all_paths = json.load(f)
    paths = all_paths[task]
    return paths


def concordance(word, language, task):
    key = 'words'+str(language)
    vocab_file = read_paths(task)[key]
    print('\n\n\n',vocab_file)

    with open(vocab_file, 'r') as f:
        vocab = json.load(f)
        con = vocab[word]
    con_formatted = '<br><br>'.join(con)+'<br><br><br>'
    return con_formatted


def setup_to_page(uid, task):
    setup_file = read_paths(task)['setup']
    # setup_file2 = read_paths(task)['setup2']

    with open(setup_file, 'r') as f:
        setup = json.load(f)

    # if os.path.exists(setup_file2):
        # with open(setup_file2, 'r') as f:
            # setup2 = json.load(f)
        # setup['queries'].append(setup2['queries'])
        # setup['nn1'].append(setup2['nn1'])
        # setup['nn2'].append(setup2['nn2'])
        # setup['lang'].append(setup2['lang'])

    n_queries = len(setup['queries'])
    for i in range(n_queries):
        page = Page(
            page_num = i,
            keyword = setup['queries'][i],
            lang = setup['lang'][i],
            nn1 = json.dumps(setup['nn1'][i]),
            nn2 = json.dumps(setup['nn2'][i]),
            user_id = uid,
        )
        db.session.add(page)
    db.session.commit()


def load_vocab(language, task):
    key = 'words'+str(language)
    vocab_file = read_paths(task)[key]
    with open(vocab_file, 'r') as f:
        vocab = json.load(f)
    vocab_list = list(vocab)
    return vocab_list



def load_page(user, entry):
    page = user.pages.filter_by(page_num=entry).first()
    row = {}
    row['lang'] = page.lang
    row['query'] = page.keyword
    row['nn1'] = json.loads(page.nn1)
    row['nn2'] = json.loads(page.nn2)
    return row





