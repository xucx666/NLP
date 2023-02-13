import argparse
from pathlib import Path

CURPATH = Path(__file__).resolve()
ROOT = CURPATH.parents[1]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=ROOT / 'classification/data', help='path to data')
    parser.add_argument('--bert_dir', type=str, default=ROOT / 'pretrained_model', help='path to bert')
    parser.add_argument('--checkpoint_dir', type=str, default=ROOT / 'classification/model', help='path to bert')
    parser.add_argument('--model_save_name', type=str, default='bert_cls', help='path to save model')
    parser.add_argument('--log_save_name', type=str, default='bert_model_log', help='path to log')
    parser.add_argument('--class_num', type=int, default=15, help='number of label')
    parser.add_argument('--batch_size', type=int, default=8, help='batch_size')
    parser.add_argument('--dropout', type=float, default=0.5, help='')
    parser.add_argument('--feature_num', type=int, default=768, help='output size of bert')
    parser.add_argument('--fine_tune', type=str, default=False, help='whether is finetune bert')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--max_epoch', type=int, default=50, help='epoch')
    parser.add_argument('--alpha', type=float, default=1.0, help='alpha')
    parser.add_argument('--mixup', type=str, default=False, help='mix up')
    parser.add_argument('--mixup_method', type=str, default='sent', help='mix up method')
    parser.add_argument('--period', type=int, default=200, help='print period')
    parser.add_argument('--test_epoch', type=int, default=1, help='test epoch')
    return parser.parse_args()
