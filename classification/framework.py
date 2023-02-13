import os
import time
import torch
import data_loader

import torch.nn as nn
import numpy as np
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score


class Framework(object):
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cpu')
        self.loss_function = nn.CrossEntropyLoss(reduction='none')

    def logging(self, s, print_=True, log_=True):
        if print_:
            print(s)
        if log_:
            with open(os.path.join(self.config.data_dir, self.config.log_save_name), 'a+') as f_log:
                f_log.write(s + '\n')

    def train(self, model_pattern):
        model = model_pattern(self.config)
        model.to(self.device)

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=self.config.learning_rate)

        # # whether use multi gpu:
        # if self.config.multi_gpu:
        #     model = nn.DataParallel(ori_model)
        # else:
        #     model = ori_model

        # check the model dir
        if not os.path.exists(self.config.checkpoint_dir):
            os.mkdir(self.config.checkpoint_dir)

        # check the log dir
        if not os.path.exists(self.config.data_dir):
            os.mkdir(self.config.data_dir)

        # training data
        train_data_loader = data_loader.get_loader(self.config, 'train')
        # dev data
        dev_data_loader = data_loader.get_loader(self.config, 'dev')

        # other
        model.train()
        global_step = 0
        loss_sum = 0

        best_f1_score = 0
        best_precision = 0
        best_recall = 0

        best_epoch = 0
        init_time = time.time()
        start_time = time.time()

        # the training loop
        for epoch in range(self.config.max_epoch):
            epoch_start_time = time.time()

            for i, batch in enumerate(train_data_loader):
                (input_ids, input_masks, segment_ids, labels) = tuple(b.to(self.device) for b in batch)
                if self.config.mixup and epoch <= 0.7 * self.config.max_epoch:
                    batch_size = input_ids.shape[0]
                    index = torch.randperm(batch_size)
                    (input_ids2, input_masks2, segment_ids2, labels2) = (
                        input_ids[index], input_masks[index], segment_ids[index], labels[index])
                    lam = np.random.beta(self.config.alpha, self.config.alpha)
                    if self.config.mixup_method == 'sent':
                        pred = model.forward_mix_sent(x1=input_ids, token_type1=segment_ids, att1=input_masks,
                                                      x2=input_ids2, token_type2=segment_ids2, att2=input_masks2,
                                                      lam=lam)
                    elif self.config.mixup_method == 'embed':
                        pred = model.forward_mix_embed(x1=input_ids, token_type1=segment_ids, att1=input_masks,
                                                       x2=input_ids2, token_type2=segment_ids2, att2=input_masks2,
                                                       lam=lam)
                    else:
                        raise ('unseen mixup method')
                    loss = lam * self.loss_function(pred, labels) + (1 - lam) * self.loss_function(pred, labels2)

                else:
                    pred = model(input_ids=input_ids, token_type_ids=segment_ids, attn_masks=input_masks)
                    loss = self.loss_function(pred, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                global_step += 1
                loss_sum += loss.item()

                if global_step % self.config.period == 0:
                    cur_loss = loss_sum / self.config.period
                    elapsed = time.time() - start_time
                    self.logging("epoch: {:3d}, step: {:4d}, speed: {:5.2f}ms/b, train loss: {:5.3f}".
                                 format(epoch, global_step, elapsed * 1000 / self.config.period, cur_loss))
                    loss_sum = 0
                    start_time = time.time()

            print("total time {}".format(time.time() - epoch_start_time))

            # if (epoch + 1) % self.config.test_epoch == 0:
            if epoch == 0 or (epoch + 1) % self.config.test_epoch == 0:
                eval_start_time = time.time()
                model.eval()
                # call the test function
                precision, recall, f1_score = self.test(dev_data_loader, model)

                self.logging('epoch {:3d}, eval time: {:5.2f}s, f1: {:4.3f}, precision: {:4.3f}, recall: {:4.3f}'.
                             format(epoch, time.time() - eval_start_time, f1_score, precision, recall))

                if f1_score > best_f1_score:
                    best_f1_score = f1_score
                    best_epoch = epoch
                    best_precision = precision
                    best_recall = recall
                    self.logging(
                        "saving the model, epoch: {:3d}, precision: {:4.3f}, recall: {:4.3f}, best f1: {:4.3f}".
                        format(best_epoch, best_precision, best_recall, best_f1_score))
                    # save the best model

                    path = os.path.join(self.config.checkpoint_dir, self.config.model_save_name)
                    if not self.config.debug:
                        torch.save(model.state_dict(), path)
                model.train()

        self.logging("finish training")
        self.logging("best epoch: {:3d}, precision: {:4.3f}, recall: {:4.3}, best f1: {:4.3f}, total time: {:5.2f}s".
                     format(best_epoch, best_precision, best_recall, best_f1_score, time.time() - init_time))

    def test(self, test_data_loader, model):
        s = time.time()
        true_labels = np.array([])
        predict_labels = np.array([])
        with torch.no_grad():
            for eval_step, dev_batch in enumerate(test_data_loader):
                print(eval_step)
                dev_batch = tuple(t.to(self.device) for t in dev_batch)
                input_ids, input_masks, segment_ids, label_ids = dev_batch
                true_labels = np.append(true_labels, label_ids.data.cpu().numpy())
                out = model(input_ids=input_ids, token_type_ids=segment_ids, attn_masks=input_masks)
                out = nn.Softmax(dim=-1)(out)
                _, pred = torch.max(out, 1)
                predict_labels = np.append(predict_labels, pred.cpu().numpy())

        print("test time {}".format(time.time() - s))
        p = precision_score(true_labels, predict_labels, average='micro')
        r = recall_score(true_labels, predict_labels, average='micro')
        f1 = f1_score(true_labels, predict_labels, average='micro')
        return p, r, f1
