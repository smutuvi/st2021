import os
import logging
from tqdm import tqdm, trange
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, ConcatDataset, TensorDataset
from transformers import BertConfig, AdamW, get_linear_schedule_with_warmup
import copy
import math
from model import RBERT
from utils import set_seed, write_f1_tc, write_prediction_re, write_prediction_tc, write_prediction_wic, compute_metrics, get_label, MODEL_CLASSES, WiCMODEL_CLASSES, ReMODEL_CLASSES, ContrastiveLoss, SoftContrastiveLoss

logger = logging.getLogger(__name__)

class Trainer(object):
    def __init__(self, args, train_dataset = None, dev_dataset = None, test_dataset = None, labelset = None, unlabeled = None, \
                num_labels = 11, id2label = None, label2id = None, data_size = 100):
                #masked_train_dataset = None, masked_dev_dataset = None,  masked_test_dataset = None, masked_unlabeled_dataset = None):
        self.args = args
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset
        self.unlabeled = unlabeled
        self.data_size = data_size

        self.label_lst = labelset
        self.num_labels = num_labels
        self.id2label = id2label
        self.label2id = label2id

        self.w = args.soft_label_weight
        self.k = (1-self.w)/(self.num_labels-1)
        self.label_matrix = torch.eye(self.num_labels) * (self.w - self.k) + self.k * torch.ones(self.num_labels)

        # if args.task_type == 'wic':
        #     self.config_class, self.model_class, _ = WiCMODEL_CLASSES[args.model_type]
        # elif args.task_type == 're':
        #     self.config_class, self.model_class, _ = ReMODEL_CLASSES[args.model_type]
        # else:
        #     self.config_class, self.model_class, _ = MODEL_CLASSES[args.model_type]
        
        self.config_class, self.model_class, _ = MODEL_CLASSES[args.model_type]
        self.bert_config = self.config_class.from_pretrained(args.model_name_or_path, num_labels=self.num_labels, finetuning_task=args.task)
        
        self.model = self.model_class(self.bert_config, args)
        self.init_model()
        #self.model.to(self.device)

    def init_model(self):
        # GPU or CPU
        self.device = "cuda" if torch.cuda.is_available() and not self.args.no_cuda else "cpu"
        self.model = nn.DataParallel(self.model)
        self.model = self.model.to(self.device)


    def train(self):
        # if self.args.method == 'clean':
        #     print('clean data!')
        #     concatdataset = ConcatDataset([self.train_dataset, self.unlabeled])
        #     train_sampler = RandomSampler(concatdataset)
        #     train_dataloader = DataLoader(concatdataset, sampler=train_sampler, batch_size = self.args.batch_size)
        # else:
        #     train_sampler = RandomSampler(self.train_dataset)
        #     train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.batch_size)
        
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.batch_size)
    
        #assert 0
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total)


        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Total train batch size = %d", self.args.batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)
        global_step = 0
        tr_loss = 0.0
        self.model.zero_grad()

        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")
        set_seed(self.args)  # Added here for reproductibility (even between python 2 and 3)
        
        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")
        set_seed(self.args)
        criterion = nn.KLDivLoss(reduction = 'batchmean')

        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU
                input_ids, input_mask, segment_ids, label_ids, valid_ids,l_mask = batch

                inputs = {'input_ids':      batch[0],
                        'attention_mask': batch[1],
                        'token_type_ids': batch[2],
                        'labels':         batch[3]}
                # outputs = self.model(**inputs)
                # loss1 = outputs[0]
                # logits = outputs[1]
                outputs = self.model(input_ids, segment_ids, input_mask, label_ids,valid_ids,l_mask)
                loss = outputs #[0] 
                # loss = criterion(input = F.log_softmax(logits), target = self.label_matrix[batch[3]].to(self.device))
                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps
                if torch.cuda.device_count() > 1:
                    #print(loss.size(), torch.cuda.device_count())
                    loss = loss.mean()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                loss.backward()
                tr_loss += loss.item()
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    global_step += 1
                    epoch_iterator.set_description("iteration:%d, w=%.1f, Loss:%.3f" % (_, self.args.soft_label_weight, tr_loss/global_step))
                    if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                        self.evaluate('dev', global_step)
                        self.evaluate('test', global_step)
                    if self.args.save_steps > 0 and global_step % self.args.save_steps == 0:
                        self.save_model()
                
                if 0 < self.args.max_steps < global_step:
                    epoch_iterator.close()
                    break

            if 0 < self.args.max_steps < global_step:
                train_iterator.close()
                break
        #assert 0
        return global_step, tr_loss / global_step

    def evaluate(self, mode, global_step=-1):
        # We use test dataset because semeval doesn't have dev dataset
        if mode == 'test':
            dataset = self.test_dataset
        elif mode == 'dev':
            dataset = self.dev_dataset
        else:
            raise Exception("Only dev and test dataset available")

        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.batch_size)

        # Eval!
        logger.info("***** Running evaluation on %s dataset *****", mode)
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", self.args.batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        self.model.eval()
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {
                            'input_ids': batch[0],
                            'attention_mask': batch[1],
                            'token_type_ids': batch[2],
                            'labels': batch[3],
                          }
                if self.args.task_type=='wic':
                    inputs['keys'] = batch[6]
                elif self.args.task_type=='re':
                    inputs['e1_mask'] = batch[4]
                    inputs['e2_mask'] = batch[5]
                outputs = self.model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(
                    out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        results = {
            "loss": eval_loss
        }
        preds = np.argmax(preds, axis=1)
        if self.args.task_type == 're':
            write_prediction_re(self.args, os.path.join(self.args.eval_dir, "pred/proposed_answers.txt"), preds)
        elif self.args.task_type == 'tc':
            write_prediction_tc(self.args, os.path.join(self.args.eval_dir, "pred/pred_%s_%s_%s_%d.txt"%(self.args.task, mode, self.args.method, global_step)), preds, self.id2label)
        elif self.args.task_type == 'wic':
            write_prediction_wic(self.args, os.path.join(self.args.eval_dir, "pred/pred_%s_%s_%s_%s.txt"%(self.args.task, mode, self.args.method, str(global_step))), preds, self.id2label)
        else:
            pass
        result = compute_metrics(preds, out_label_ids)
        result.update(result)

        logger.info("***** Eval results *****")

        print('Macro F1: %.4f, Micro F1: %.4f, Accu: %.4f'%(result["macro-f1"], result["micro-f1"], result["acc"]))
        write_f1_tc(self.args, os.path.join(self.args.eval_dir, "pred_%s_%s"%(self.args.task, mode)), result["macro-f1"], result["micro-f1"], result["acc"],global_step)
  
        return results

    def save_model(self):
        # Save model checkpoint (Overwrite)
        output_dir = os.path.join(self.args.model_dir)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, 'training_config.bin'))
        logger.info("Saving model checkpoint to %s", output_dir)

    def load_model(self):
        # Check whether model exists
        if not os.path.exists(self.args.model_dir):
            raise Exception("Model doesn't exists! Train first!")

        try:
            self.bert_config = self.config_class.from_pretrained(self.args.model_dir)
            logger.info("***** Config loaded *****")
            self.model = self.model_class.from_pretrained(self.args.model_dir, config=self.bert_config, args=self.args)
            self.model.to(self.device)
            logger.info("***** Model Loaded *****")
        except:
            raise Exception("Some model files might be missing...")
         