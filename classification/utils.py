import os


def load_data_info(args, examples, labels, mode):
    out_file = os.path.join(args.data_dir, '{}_info.txt'.format(mode))
    results = [0] * (args.class_num + 1)  # 各标签的个数和数据集中文本最长长度
    label2id = {name: i for i, name in enumerate(labels)}
    print(label2id)
    if os.path.exists(out_file):
        return '文件信息已经记下了，其路径为{}'.format(out_file)
    else:
        for example in examples:
            idx = example.label
            results[idx] += 1
            results[-1] = max(results[-1], len(example.text))
        with open(out_file, 'a', encoding='utf_8_sig') as f:
            for label in labels:
                idx = label2id[label]
                f.write('标签{}的数量为:{}\n'.format(label, results[idx]))
            f.write('文本中最长文本长度为:{}'.format(results[-1]))
    return results
