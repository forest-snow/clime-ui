import json
import logging
import os

import torch
from torchtext.data import Field, LabelField, TabularDataset, BucketIterator


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