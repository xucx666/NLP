import torch.optim as optim
from torch import nn
import os
import data_loader

import torch
import numpy as np
import json
import time


class Framework(object):
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cpu')
        self.loss_function = nn.CrossEntropyLoss(reduction='none')

    def logging(self, s, print_=True, log_=True):
        if print_:
            print(s)
        if log_:
            with open(os.path.join(self.config.log_dir, self.config.log_save_name), 'a+') as f_log:
                f_log.write(s + '\n')

    def train(self, model_pattern):
        model = model_pattern(self.config)
        model.to(self.device)

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=self.config.learning_rate)

        # whether use multi gpu:
        if self.config.multi_gpu:
            model = nn.DataParallel(model)
        else:
            model = model

        # define the loss function

        # check the check_point dir
        if not os.path.exists(self.config.checkpoint_dir):
            os.mkdir(self.config.checkpoint_dir)

        # check the log dir
        if not os.path.exists(self.config.data_dir):
            os.mkdir(self.config.data_dir)

        # training data
        train_data_loader = data_loader.get_loader(self.config, 'train', num_workers=2)
        # dev data
        test_data_loader = data_loader.get_loader(self.config, 'dev', num_workers=2)

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

                pred = model(input_ids=input_ids, token_type_id=segment_ids, attn_masks=input_masks)
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

            # if epoch > 20 and (epoch + 1) % self.config.test_epoch == 0:
            if (epoch + 1) % self.config.test_epoch == 0:
                eval_start_time = time.time()
                model.eval()
                # call the test function
                precision, recall, f1_score = self.test(test_data_loader, model)

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
        pass
