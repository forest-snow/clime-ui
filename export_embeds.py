"""Convert binarized embeddings to text files."""

from argparse import ArgumentParser
import json
import torch

from utils import load_embeds


def main():
    parser = ArgumentParser()
    parser.add_argument('embed_dir', help='binarized embedding directory')
    parser.add_argument('output', help='output file')
    parser.add_argument('--prefix', default='', help='language prefix')
    args = parser.parse_args()

    E, word2id = load_embeds(args.embed_dir)
    with open(args.output, 'w') as f:
        for w in word2id:
            print(w, ' '.join(str(float(x)) for x in E[word2id[w], :]), file=f)


if __name__ == '__main__':
    main()