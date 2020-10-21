"""Find most important words from a model by gradient."""

from argparse import ArgumentParser
import logging
import numpy as np
import torch
from torch import nn

from utils import load_data
from model import get_model, evaluate_rank


def get_idf(iterator, n_words):
    """Compute inverse document frequency (IDF) of a word type."""
    n_docs = 0
    doc_freqs = torch.ones(n_words)
    for batch in iterator:
        _, n_sent = batch.text[0].shape
        n_docs += n_sent
        for i in range(n_sent):
            words = set(batch.text[0][:, i])
            for w in words:
                doc_freqs[w] += 1
    idf = -torch.log(doc_freqs / n_docs)
    return idf


def get_ranking(values, vocab):
    """rank words based on descending order of values"""
    ranking = np.argsort(-values.cpu())
    values_ranked = values[ranking]
    words_ranked = [vocab.itos[i] for i in ranking]
    return words_ranked, values_ranked


def rank_vocab(model, data_iterator, vocab, device):
    """rank words by global salience."""
    embed_grad = model.embeddings.weight.grad
    grad_words = torch.norm(embed_grad, p=2, dim=1)
    idf = get_idf(data_iterator, grad_words.size()[0]).to(device)
    grad_words_idf = torch.mul(grad_words, idf)

    # retrieve ranking based on textual saliency
    return get_ranking(grad_words_idf, vocab)


def main():
    logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
    parser = ArgumentParser()
    parser.add_argument('model', help='model checkpoint')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='mini-batch size')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA')
    args = parser.parse_args()
    logging.info(vars(args))

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    checkpoint = torch.load(args.model, map_location=device)
    model = get_model(checkpoint['type'], checkpoint['vocab_size'],
                      checkpoint['emb_dim'], checkpoint['n_classes']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    min_length = 5 if checkpoint['type'] == 'cnn' else -1  # padding for CNN
    (data_iter, _, _), vocab, n_classes = load_data(
        (checkpoint['train'], checkpoint['dev'], checkpoint['test']),
        batch_size=args.batch_size,
        min_length=min_length,
        device=device
    )

    criterion = nn.CrossEntropyLoss().to(device)
    evaluate_rank(model, data_iter, criterion)
    words, scores = rank_vocab(model, data_iter, vocab, device)
    for word, score in zip(words, scores):
        print('{}\t{:0.5f}'.format(word, score))


if __name__ == '__main__':
    main()