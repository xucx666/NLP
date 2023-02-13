import torch
import os
import numpy as np
from tqdm import tqdm
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from utils import load_data_info
from get_data import LABEL
from args import get_args

MAX_LENGTH = 128


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, uid, text, label=None):
        self.uid = uid
        self.text = text
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_csv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        # dicts = []
        data = pd.read_csv(input_file)
        return data


class MyProcessor(DataProcessor):
    '''自定义数据读取方法，针对csv文件

    Returns:
        examples: 数据集，包含index、中文文本、类别三个部分
    '''

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, 'train.csv')), 'train')

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, 'dev.csv')), 'dev')

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, 'test.csv')), 'test')

    def get_labels(self):
        labels = []
        for l in LABEL:
            name = l[1]
            labels.append(name)
        return labels

    def _create_examples(self, data, set_type):
        examples = []
        labels = self.get_labels()
        label2id = {name: i for i, name in enumerate(labels)}
        for index, row in data.iterrows():
            guid = "%s-%s" % (set_type, index)
            text = row['text']
            label = label2id[row['label']]
            examples.append(
                InputExample(uid=guid, text=text, label=label))
        return examples


def convert_examples_to_features(examples, tokenizer, max_seq_length=MAX_LENGTH):
    '''Loads a data file into a list of `InputBatch`s.

    Args:
        examples      : [List] 输入样本，句子和label
        label_list    : [List] 所有可能的类别，0和1
        max_seq_length: [int] 文本最大长度
        tokenizer     : [Method] 分词方法

    Returns:
        features:
            input_ids  : [ListOf] token的id，在chinese模式中就是每个分词的id，对应一个word vector
            input_mask : [ListOfInt] 真实字符对应1，补全字符对应0
            segment_ids: [ListOfInt] 句子标识符，第一句全为0，第二句全为1
            label_id   : [ListOfInt] 将Label_list转化为相应的id表示
    '''
    features = []
    for (ex_index, example) in enumerate(examples):
        input_ids, input_mask, segment_ids = None, None, None
        # 分词
        tokens = tokenizer.tokenize(example.text)
        # tokens进行编码

        # 需要裁剪
        if len(tokens) > max_seq_length - 2:
            tokens = tokens[:max_seq_length - 2]
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        segment_ids = [0] * len(input_ids)

        assert len(input_ids) == len(input_mask) and len(input_mask) == len(segment_ids)

        label_id = example.label

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))
    return features


class MyDataset(Dataset):
    def __init__(self, features):
        self.nums = len(features)
        self.input_ids = [f.input_ids for f in features]
        self.input_mask = [f.input_mask for f in features]
        self.segment_ids = [f.segment_ids for f in features]

        self.label_id = None

        self.label_id = [f.label_id for f in features]

    def __getitem__(self, index):
        data = {'input_ids': self.input_ids[index],
                'input_masks': self.input_mask[index],
                'segment_ids': self.segment_ids[index],
                'label_id': None
                }

        if self.label_id[index] is not None:
            data['label_id'] = self.label_id[index]

        return data

    def __len__(self):
        return self.nums


def pad_batch(batchs, pad_value=0):
    max_seq_length = max([len(sample['input_ids']) for sample in batchs])
    input_ids = torch.tensor(
        [sample['input_ids'] + [pad_value] * (max_seq_length - len(sample['input_ids'])) for sample in batchs])
    input_masks = torch.tensor(
        [sample['input_masks'] + [pad_value] * (max_seq_length - len(sample['input_ids'])) for sample in batchs])
    segment_ids = torch.tensor(
        [sample['segment_ids'] + [pad_value] * (max_seq_length - len(sample['input_ids'])) for sample in
         batchs])
    labels = torch.tensor([sample['label_id'] for sample in batchs if sample is not None])
    return (input_ids, input_masks, segment_ids, labels)


def get_loader(args, mode, num_workers=2):
    tokenizer = BertTokenizer.from_pretrained(os.path.join(args.bert_dir, 'vocab.txt'))
    processer = MyProcessor()
    if mode == 'train':
        examples = processer.get_train_examples(args.data_dir)
        results = load_data_info(args, examples, processer.get_labels(), 'train')
        print(results)

    elif mode == 'dev':
        examples = processer.get_dev_examples(args.data_dir)
    else:
        examples = processer.get_test_examples(args.data_dir)
    features = convert_examples_to_features(examples, tokenizer)
    dataset = MyDataset(features)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=pad_batch,
                             num_workers=num_workers)
    return data_loader


if __name__ == '__main__':
    args = get_args()
    print(args)
    data_loader = get_loader(args, 'train')
    for i, batch in enumerate(data_loader):
        print(batch)
