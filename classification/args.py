import argparse
from pathlib import Path

CURPATH = Path(__file__).resolve()
ROOT = CURPATH.parents[1]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=ROOT / 'classification/data', help='path to data')
    parser.add_argument('--bert_dir', type=str, default=ROOT / 'pretrained_model', help='path to bert')
    parser.add_argument('--class_num', type=int, default=15, help='number of label')
    parser.add_argument('--batch_size', type=int, default=8, help='batch_size')
    return parser.parse_args()
