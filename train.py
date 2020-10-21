"""Train a model."""

from argparse import ArgumentParser
import json
import logging
import numpy as np
import torch
from torch import nn, optim

from utils import load_embeds, load_data
from model import get_model, train, evaluate


def main():
    logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

    parser = ArgumentParser()
    parser.add_argument('--src-emb', required=True, help='source language embedding directory')
    parser.add_argument('--tgt-emb', required=True, help='target language embedding directory')
    parser.add_argument('--train', required=True, help='train set')
    parser.add_argument('--dev', help='dev set')
    parser.add_argument('--test', required=True, help='test set')
    parser.add_argument('--type', choices=['cnn', 'dan', 'lr'], default='cnn',
                        help='type of model (cnn, dan, lr)')
    parser.add_argument('--output', help='output path to save model')
    parser.add_argument('--dropout', type=float, default=.0, help='dropout rate')
    parser.add_argument('--batch-size', type=int, default=100, help='mini-batch size')
    parser.add_argument('--epoch', type=int, default=30, help='number of epoch')
    parser.add_argument('--seed', type=int, default=31, help='random seed')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA')
    parser.add_argument('--runs', type=int, default=10, help='number of runs')
    args = parser.parse_args()
    if args.dev is None:
        args.dev = args.train
    logging.info(vars(args))
    checkpoint = vars(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    # load embeddings
    logging.info('Load embeddings.')
    E_src, words_src = load_embeds(args.src_emb)
    E_tgt, words_tgt = load_embeds(args.tgt_emb)

    # merge source and target word embeddings and vocabularies
    E = torch.cat((E_src, E_tgt))
    n_src_words = len(words_src)
    words = words_src
    for w in words_tgt:
        words[w] = words_tgt[w] + n_src_words

    # load dataset
    logging.info('Load dataset and vocabulary.')
    min_length = 5 if args.type == 'cnn' else -1  # minimum padding for CNN
    (train_iter, dev_iter, test_iter), vocab, n_classes = load_data(
        (args.train, args.dev, args.test),
        batch_size=args.batch_size,
        min_length=min_length,
        device=device
    )
    vocab.set_vectors(words, E, E.size()[1])

    # use vocab.vectors instead of E to handle UNK.
    vocab_size, emb_dim = vocab.vectors.size()
    logging.info('Vocabulary size: %d' % vocab_size)
    logging.info('Word embedding dimension: %d' % emb_dim)
    checkpoint['vocab_size'] = vocab_size
    checkpoint['emb_dim'] = emb_dim
    checkpoint['n_classes'] = n_classes

    test_accs = []
    for n_run in range(args.runs):
        logging.info('Run %d' % n_run)
        # initialize and freeze embeddings
        model = get_model(args.type, vocab_size, emb_dim, n_classes,
                          dropout=args.dropout).to(device)
        model.embeddings.weight.data.copy_(vocab.vectors)
        model.embeddings.weight.requires_grad = False

        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = optim.Adam(model.parameters())
        best_acc = 0.0
        logging.info('Start training')
        for n_epoch in range(args.epoch):
            train_loss, train_acc = train(model, train_iter, optimizer, criterion)
            dev_loss, dev_acc = evaluate(model, dev_iter, criterion)
            test_loss, test_acc = evaluate(model, test_iter, criterion)
            logging.info('Epoch {} | Test acc: {:.4f}'.format(n_epoch, test_acc))
            # save progress when accuracy increases
            if dev_acc > best_acc:
                best_acc = dev_acc
                final_test_acc = test_acc
                checkpoint['n_epoch'] = n_epoch
                checkpoint['model_state_dict'] = model.state_dict()
                checkpoint['optim_state_dict'] = optimizer.state_dict()
                if args.output is not None:
                    torch.save(checkpoint, args.output)
        logging.info('Test acc of final model: {:.4f}'.format(final_test_acc))
        test_accs.append(final_test_acc)
    avg_test_acc = sum(test_accs) / len(test_accs)
    logging.info('Average test acc over {} runs: {:.4f}'.format(args.runs, avg_test_acc))


if __name__ == '__main__':
    main()