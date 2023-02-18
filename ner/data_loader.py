import torch
import os
import re
import json
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
from args import get_args

MAX_LENGTH = 512


class NerDataset(Dataset):
    def __init__(self, args, mode):
        super(NerDataset, self).__init__()
        self.file = os.path.join(args.data_dir, '{}_pre.txt'.format(mode))
        self.texts, self.labels = [], []
        self.tokenizer = BertTokenizer.from_pretrained(os.path.join(args.bert_dir, 'vocab.txt'))
        self.tag2id = self.get_tag(args)
        with open(self.file, 'r', encoding='UTF-8-sig') as f:
            i = 0
            for line in f:
                # text行
                if i % 2 == 0:
                    text = line.rstrip('\n')
                    self.texts.append(text)
                else:
                    label = line.rstrip('\n').split(' ')
                    self.labels.append(label)
                i += 1
        self.split_long_sentence()

    def split_long_sentence(self):
        tmp_texts, tmp_labels = [], []
        for text, label in zip(self.texts, self.labels):
            if len(text) <= MAX_LENGTH - 2:
                tmp_texts.append(text)
                tmp_labels.append(label)
            else:
                start_idx = 0
                pattern = r'\.|。|！|\[|\]|'
                split_texts = re.split(pattern, text)
                for s_text in split_texts:
                    if len(s_text) <= MAX_LENGTH - 2:
                        tmp_texts.append(s_text)
                        tmp_labels.append(label[start_idx:start_idx + len(s_text)])
                        start_idx += len(s_text)
                    else:
                        print(text)
                        break

        self.texts = tmp_texts
        self.labels = tmp_labels
        assert len(self.texts) == len(self.labels)

    def get_tag(self, args):
        file = os.path.join(args.data_dir, 'tag2id.json')
        with open(file, 'r') as f:
            tag2id = json.load(f)
        return tag2id

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text, label = self.texts[idx], self.labels[idx]
        label = [self.tag2id[l] for l in label]
        tokens = [t for t in text]
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        label = [self.tag2id['O']] + label + [self.tag2id['O']]
        text_len = len(tokens)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        token_type_ids = [0] * text_len
        input_masks = [1] * text_len
        return input_ids, token_type_ids, input_masks, text_len, label, tokens

    def pad_batch(self, batchs, pad=0):
        batch_input_ids, batch_token_type_ids, batch_input_masks, batch_text_len, batch_labels, batch_tokens = zip(
            *batchs)
        max_text_len = max(batch_text_len)
        batch_input_ids = torch.LongTensor(
            [input_ids + [pad] * (max_text_len - len(input_ids)) for input_ids in batch_input_ids])
        batch_token_type_ids = torch.LongTensor(
            [token_type_ids + [pad] * (max_text_len - len(token_type_ids)) for token_type_ids in batch_token_type_ids])
        batch_input_masks = torch.LongTensor(
            [input_masks + [pad] * (max_text_len - len(input_masks)) for input_masks in batch_input_masks])
        batch_labels = torch.LongTensor(
            [label + [self.tag2id['O']] * (max_text_len - len(label)) for label in batch_labels])
        return batch_input_ids, batch_token_type_ids, batch_input_masks, batch_labels, batch_tokens


def get_loader(args, mode):
    dataset = NerDataset(args, mode)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.pad_batch)
    return dataloader


if __name__ == '__main__':
    args = get_args()
    dataloader = get_loader(args, 'train')
    for i, batch in enumerate(dataloader):
        batch_input_ids, batch_token_type_ids, batch_input_masks, batch_labels, batch_tokens = batch
        print(i)
        assert batch_labels.shape == batch_input_ids.shape == batch_token_type_ids.shape == batch_input_masks.shape
