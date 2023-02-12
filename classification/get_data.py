# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

CURPATH = Path(__file__).resolve()
PARENT = CURPATH.parents[0]

LABEL = [
    [100, '故事', 'news_story'],
    [101, '文化', 'news_culture'],
    [102, '娱乐', 'news_entertainment'],
    [103, '体育', 'news_sports'],
    [104, '财经', 'news_finance'],
    # [105, '时政 新时代', 'nineteenth'],
    [106, '房产', 'news_house'],
    [107, '汽车', 'news_car'],
    [108, '教育', 'news_edu'],
    [109, '科技', 'news_tech'],
    [110, '军事', 'news_military'],
    # [111 宗教 无，凤凰佛教等来源],
    [112, '旅游', 'news_travel'],
    [113, '国际', 'news_world'],
    [114, '股票', 'stock'],
    [115, '三农', 'news_agriculture'],
    [116, '游戏', 'news_game']
]


def get_label_dict():
    d = {}
    for l in LABEL:
        label_id = l[0]
        name = l[1]
        d[label_id] = name
    return d


def get_data():
    label2id = get_label_dict()
    texts, labels = [], []
    n = 0
    with open('toutiao_cat_data.txt', 'r', encoding='utf-8') as fp:
        ll = fp.readlines()
        g_count = len(ll)
        for l in ll:
            data = l.split('_!_')
            label = int(data[1])
            text = data[3]
            if label is not None and text is not None:
                n += 1
                texts.append(text)
                labels.append(label2id[label])
        print('load cache done, ', n)
    return texts, labels


def save_to_csv(data, name, out_path, mode):
    save_name = os.path.join(out_path, '{}.csv'.format(mode))
    data = pd.DataFrame(columns=name, data=data)
    data.to_csv(save_name, encoding='utf_8_sig', index=False)


def split_train_dev_test(train_ratio=0.7, dev_ratio=0.15, test_ratio=0.15):
    texts, labels = get_data()
    out_path = PARENT / 'data'
    if not os.path.exists(out_path):
        os.mkdir(out_path)
        idx = [i for i in range(len(texts))]
        train, middle = train_test_split(idx, train_size=train_ratio, test_size=dev_ratio + test_ratio)
        ratio = dev_ratio / (1 - test_ratio)
        dev, test = train_test_split(middle, test_size=ratio)
        texts, labels = np.array(texts), np.array(labels)
        train_texts, dev_texts, test_texts = texts[train], texts[dev], texts[test]
        train_labels, dev_labels, test_labels = labels[train], labels[dev], labels[test]
        train_data, dev_data, test_data = list(zip(train_texts, train_labels)), list(zip(dev_texts, dev_labels)), \
                                          list(zip(test_texts, test_labels))
        name = ['text', 'label']
        save_to_csv(train_data, name, out_path, 'train')
        save_to_csv(dev_data, name, out_path, 'dev')
        save_to_csv(test_data, name, out_path, 'test')


if __name__ == '__main__':
    split_train_dev_test()
