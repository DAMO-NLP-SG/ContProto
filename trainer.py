# encoding: utf-8


import argparse
import os
# from collections import namedtuple
from typing import Dict

import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from tokenizers import BertWordPieceTokenizer,ByteLevelBPETokenizer, Tokenizer
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import AdamW
from torch.optim import SGD
import torch.nn.functional as F

from dataloaders.dataload import BERTNERDataset
from dataloaders.truncate_dataset import TruncateDataset
from dataloaders.collate_functions import collate_to_max_length
from models.bert_model_spanner import BertNER
from models.config_spanner import BertNerConfig, XLMRNerConfig
# from utils.get_parser import get_parser
from radom_seed import set_random_seed
from eval_metric import span_f1,span_f1_prune,get_predict,get_predict_prune, get_pruning_predIdxs
import random
import logging
logger = logging.getLogger(__name__)
#set_random_seed(0)

import pickle
import numpy as np


class BertNerTagger(pl.LightningModule):
    """MLM Trainer"""

    def __init__(
        self,
        args: argparse.Namespace
    ):
        """Initialize a model, tokenizer and config."""
        super().__init__()
        if isinstance(args, argparse.Namespace):
            self.save_hyperparameters(args)
            self.args = args
        else:
            # eval mode
            TmpArgs = namedtuple("tmp_args", field_names=list(args.keys()))
            self.args = args = TmpArgs(**args)

        self.bert_dir = args.bert_config_dir
        self.data_dir = self.args.data_dir

        # bert_config = BertNerConfig.from_pretrained(args.bert_config_dir,
        #                                                  hidden_dropout_prob=args.bert_dropout,
        #                                                  attention_probs_dropout_prob=args.bert_dropout,
        #                                                  model_dropout=args.model_dropout)
        
        bert_config = XLMRNerConfig.from_pretrained(args.bert_config_dir,
                                                         hidden_dropout_prob=args.bert_dropout,
                                                         attention_probs_dropout_prob=args.bert_dropout,
                                                         model_dropout=args.model_dropout)

        self.model = BertNER.from_pretrained(args.bert_config_dir,
                                                  config=bert_config,
                                                  args=self.args)
        logging.info(str(args.__dict__ if isinstance(args, argparse.ArgumentParser) else args))

        self.optimizer = args.optimizer
        self.n_class = args.n_class

        self.max_spanLen = args.max_spanLen
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')
        self.classifier = torch.nn.Softmax(dim=-1)

        self.fwrite_epoch_res = open(args.fp_epoch_result, 'w')
        self.fwrite_epoch_res.write("f1, recall, precision, correct_pred, total_pred, total_golden\n")
        
        # Cont Pro
        #self.prototypes = torch.zeros(self.args.n_class, args.q_dim)
        self.register_buffer('prototypes', torch.zeros(self.args.n_class, args.q_dim))
        
        self.register_buffer('auto_margin', torch.zeros(self.args.n_class - 1))
        
        with open(self.args.load_soft, 'rb') as f:
            init_confidence_list = pickle.load(f)
        self.num_span_upperb = self.args.max_spanLen * self.args.bert_max_length
        init_confidence = torch.tensor([[sent + [[1] + [0]*(self.args.n_class-1)]*(self.num_span_upperb-len(sent))] for sent in init_confidence_list]).squeeze(1) # TODO: where is this extra dimension coming from?
        print('Dimension of init confidence is: ', init_confidence.shape)
        
        self.loss_fn_ul = partial_loss(init_confidence, postprocess_pseudo=self.args.postprocess_pseudo)
        self.loss_cont_fn = SupConLoss()
        
    

    @staticmethod
    def get_parser():
        def str2bool(v):
            if v.lower() in ('yes', 'true', 't', 'y', '1'):
                return True
            elif v.lower() in ('no', 'false', 'f', 'n', '0'):
                return False
            else:
                raise argparse.ArgumentTypeError('Boolean value expected.')

        parser = argparse.ArgumentParser(description="Training")

        # basic argument&value
        parser.add_argument("--data_dir", type=str, required=True, help="data dir")
        parser.add_argument("--bert_config_dir", type=str, required=True, help="bert config dir")
        parser.add_argument("--pretrained_checkpoint", default="", type=str, help="pretrained checkpoint path")
        parser.add_argument("--bert_max_length", type=int, default=128, help="max length of dataset")
        parser.add_argument("--batch_size", type=int, default=10, help="batch size")
        parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
        parser.add_argument("--workers", type=int, default=0, help="num workers for dataloader")
        parser.add_argument("--weight_decay", default=0.01, type=float,
                            help="Weight decay if we apply some.")
        parser.add_argument("--warmup_steps", default=0, type=int,
                            help="warmup steps used for scheduler.")
        parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                            help="Epsilon for Adam optimizer.")


        parser.add_argument("--model_dropout", type=float, default=0.2,
                            help="model dropout rate")
        parser.add_argument("--bert_dropout", type=float, default=0.2,
                            help="bert dropout rate")
        parser.add_argument("--final_div_factor", type=float, default=1e4,
                            help="final div factor of linear decay scheduler")
        parser.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw",
                            help="loss type")
        #choices=["conll03", "ace04","notebn","notebc","notewb","notemz",'notenw','notetc']
        parser.add_argument("--dataname", default="conll03",
                            help="the name of a dataset")
        parser.add_argument("--max_spanLen", type=int, default=4, help="max span length")
        # parser.add_argument("--margin", type=float, default=0.03, help="margin of the ranking loss")
        parser.add_argument("--n_class", type=int, default=5, help="the classes of a task")
        parser.add_argument("--modelName",  default='test', help="the classes of a task")

        # parser.add_argument('--use_allspan', type=str2bool, default=True, help='use all the spans with O-labels ', nargs='?',
        #                     choices=['yes (default)', True, 'no', False])

        parser.add_argument('--use_tokenLen', type=str2bool, default=False, help='use the token length (after the bert tokenizer process) as a feature',
                            nargs='?',
                            choices=['yes (default)', True, 'no', False])
        parser.add_argument("--tokenLen_emb_dim", type=int, default=50, help="the embedding dim of a span")
        parser.add_argument('--span_combination_mode', default='x,y',
                            help='Train data in format defined by --data-io param.')

        parser.add_argument('--use_spanLen', type=str2bool, default=False, help='use the span length as a feature',
                            nargs='?',
                            choices=['yes (default)', True, 'no', False])
        parser.add_argument("--spanLen_emb_dim", type=int, default=100, help="the embedding dim of a span length")

        parser.add_argument('--use_morph', type=str2bool, default=True, help='use the span length as a feature',
                            nargs='?',
                            choices=['yes (default)', True, 'no', False])
        parser.add_argument("--morph_emb_dim", type=int, default=100, help="the embedding dim of the morphology feature.")
        parser.add_argument('--morph2idx_list', type=list, help='a list to store a pair of (morph, index).', )


        parser.add_argument('--label2idx_list', type=list, help='a list to store a pair of (label, index).',)


        random_int = '%08d' % (random.randint(0, 100000000))
        print('random_int:', random_int)

        parser.add_argument('--random_int', type=str, default=random_int,help='a list to store a pair of (label, index).', )
        parser.add_argument('--param_name', type=str, default='param_name',
                            help='a prexfix for a param file name', )
        parser.add_argument('--best_dev_f1', type=float, default=0.0,
                            help='best_dev_f1 value', )
        parser.add_argument('--use_prune', type=str2bool, default=True,
                            help='best_dev_f1 value', )

        parser.add_argument("--use_span_weight", type=str2bool, default=True,
                            help="range: [0,1.0], the weight of negative span for the loss.")
        parser.add_argument("--neg_span_weight", type=float,default=0.5,
                            help="range: [0,1.0], the weight of negative span for the loss.")
        parser.add_argument("--seed", type=int, default=0)

        # ContPro
        parser.add_argument("--q_dim", type=int, default=128)
        parser.add_argument("--load_soft", type=str)
        
        parser.add_argument('--conf_ema_range', default='0.95,0.8', type=str,
                    help='pseudo target updating coefficient (phi)')
        parser.add_argument('--proto_m', default=0.99, type=float, help='momentum for computing the momving average of prototypes')
        
        parser.add_argument('--update_soft_start', default=1, type=int, help = 'Start Prototype Updating')
        
        parser.add_argument('--postprocess_pseudo', action='store_true', help = 'Whether to postprocess overlapping pseudo labels')
        parser.add_argument('--postprocess_cl', action='store_true', help = 'Whether to postprocess overlapping spans for contrastive learning')
        #parser.add_argument('--proto_margin', default=0.0, type=float, help='margin-based learning for non-entities')
        parser.add_argument('--proto_margin', default='0.0', type=str, help='margin-based learning for non-entities')
        # parser.add_argument('--update_rate', default=''1,1,1,1'', type=str, help='class specific update rate of pseudo labels')
        
        parser.add_argument('--rdrop_weight', default=0.0, type=float, help = 'rdrop weight')
        
        parser.add_argument('--cl_downsample', default=1.0, type=float, help='downsample rate of non-entity in contrastive learning')
        
        parser.add_argument('--postprocess_prot', action='store_true', help = 'whether to postprocess prototyp pseudo update ')
        
        parser.add_argument('--auto_margin', action='store_true', help = 'use auto margin')
        
        
        return parser


    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        if self.optimizer == "adamw":
            optimizer = AdamW(optimizer_grouped_parameters,
                              betas=(0.9, 0.98),  # according to RoBERTa paper
                              lr=self.args.lr,
                              eps=self.args.adam_epsilon,)
        else:
            optimizer = SGD(optimizer_grouped_parameters, lr=self.args.lr, momentum=0.9)
        num_gpus = len([x for x in str(self.args.gpus).split(",") if x.strip()])
        t_total = (len(self.train_dataloader()) // (self.args.accumulate_grad_batches * num_gpus) + 1) * self.args.max_epochs
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.args.lr, pct_start=float(self.args.warmup_steps/t_total),
            final_div_factor=self.args.final_div_factor,
            total_steps=t_total, anneal_strategy='linear'
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def forward(self, loadall,all_span_lens, all_span_idxs_ltoken,input_ids, attention_mask, token_type_ids, output_q=False):
        """"""
        return self.model(loadall,all_span_lens,all_span_idxs_ltoken,input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_q=output_q)


    def compute_loss(self,loadall, all_span_rep, span_label_ltoken, real_span_mask_ltoken,mode):
        '''

        :param all_span_rep: shape: (bs, n_span, n_class)
        :param span_label_ltoken:
        :param real_span_mask_ltoken:
        :return:
        '''
        batch_size, n_span = span_label_ltoken.size()
        all_span_rep1 = all_span_rep.view(-1,self.n_class)
        span_label_ltoken1 = span_label_ltoken.view(-1)
        loss = self.cross_entropy(all_span_rep1, span_label_ltoken1)
        loss = loss.view(batch_size, n_span)
        # print('loss 1: ', loss)
        if mode=='train' and self.args.use_span_weight: # when training we should multiply the span-weight
            span_weight = loadall[6]
            loss = loss*span_weight
            # print('loss 2: ', loss)

        loss = torch.masked_select(loss, real_span_mask_ltoken.bool())

        # print("1 loss: ", loss)
        loss= torch.mean(loss)
        # print("loss: ", loss)
        predict = self.classifier(all_span_rep) # shape: (bs, n_span, n_class)

        return loss


    def on_train_epoch_start(self):
        # Update moving average coefficient
        self.loss_fn_ul.set_conf_ema_m(self.current_epoch, self.args)

        
    def training_step(self, batch, batch_idx):
        """"""
        tf_board_logs = {
            "lr": self.trainer.optimizers[0].param_groups[0]['lr']
        }
        # tokens, token_type_ids, start_labels, end_labels, start_label_mask, end_label_mask, match_labels, sample_idx, label_idx = batch
        tokens, token_type_ids, all_span_idxs_ltoken,morph_idxs, span_label_ltoken, all_span_lens,all_span_weights,real_span_mask_ltoken,words,all_span_word,all_span_idxs = batch
        
        
        loadall = [tokens, token_type_ids, all_span_idxs_ltoken,morph_idxs, span_label_ltoken, all_span_lens,all_span_weights,
                   real_span_mask_ltoken, words, all_span_word, all_span_idxs]

        attention_mask = (tokens != 0).long()
        all_span_rep, q, last_rep = self.forward(loadall,all_span_lens,all_span_idxs_ltoken,tokens, attention_mask, token_type_ids, output_q=True)
        predicts = self.classifier(all_span_rep)
        # print('all_span_rep.shape: ', all_span_rep.shape)

        output = {}
        if self.args.use_prune:
            span_f1s,pred_label_idx = span_f1_prune(all_span_idxs, predicts, span_label_ltoken, real_span_mask_ltoken)
        else:
            span_f1s = span_f1(predicts, span_label_ltoken, real_span_mask_ltoken)
        output["span_f1s"] = span_f1s
        loss = self.compute_loss(loadall,all_span_rep, span_label_ltoken, real_span_mask_ltoken,mode='train')
        
        # Update prototypes with labeled data
        for feats, true_labels, masks in zip(q.clone().detach(), span_label_ltoken, real_span_mask_ltoken):
            for feat, true_label, mask in zip(feats, true_labels, masks):
                if mask == 1: # Excluding padding spans
                    self.prototypes[true_label] = self.prototypes[true_label]*self.args.proto_m + (1-self.args.proto_m)*feat
        # normalize prototypes    
        self.prototypes = F.normalize(self.prototypes, p=2, dim=1)
        
        
        # Unlabeled data
        ul_tokens, ul_token_type_ids, ul_all_span_idxs_ltoken,ul_morph_idxs, ul_span_label_ltoken, \
        ul_all_span_lens,ul_all_span_weights,ul_real_span_mask_ltoken,ul_words, \
        ul_all_span_word,ul_all_span_idxs, unlabel_guid = [t.to(self.device) if torch.is_tensor(t) else t for t in next(self.unlabel_loader)]
        
        ul_loadall = [ul_tokens, ul_token_type_ids, ul_all_span_idxs_ltoken,ul_morph_idxs, ul_span_label_ltoken, ul_all_span_lens,ul_all_span_weights,ul_real_span_mask_ltoken,ul_words,ul_all_span_word,ul_all_span_idxs]
        
        ul_attention_mask = (ul_tokens != 0).long()
        
        ul_all_span_rep, ul_q, ul_last_rep = self.forward(ul_loadall,ul_all_span_lens,ul_all_span_idxs_ltoken,ul_tokens, ul_attention_mask, ul_token_type_ids, output_q=True)
        ul_logits = self.classifier(ul_all_span_rep)
        
        predicted_scores = torch.softmax(ul_all_span_rep, dim=-1)
        max_scores, pseudo_labels = torch.max(predicted_scores, dim=-1)
        
        # update momentum prototypes with pseudo labels
        for feats, pred_labels, masks in zip(ul_q.clone().detach(), pseudo_labels.clone().detach(), ul_real_span_mask_ltoken):
            for feat, pred_label, mask in zip(feats, pred_labels, masks):
                if mask == 1: # Excluding padding spans
                    self.prototypes[pred_label] = self.prototypes[pred_label]*self.args.proto_m + (1-self.args.proto_m)*feat
        # normalize prototypes    
        self.prototypes = F.normalize(self.prototypes, p=2, dim=1)

        # compute prototypical logits
        prototypes_detached = self.prototypes.clone().detach()
        logits_prot = torch.matmul(ul_q, prototypes_detached.t())
        score_prot = torch.softmax(logits_prot, dim=-1)
        # analyze pseudo distribution
        output['logits_prot'] = logits_prot[ul_real_span_mask_ltoken.bool()]
        output['score_prot'] = score_prot[ul_real_span_mask_ltoken.bool()]
        #output['unlabel_pred_label'] = pseudo_labels[ul_real_span_mask_ltoken.bool()]
        output['unlabel_pred_label'] = torch.max(self.loss_fn_ul.confidence[unlabel_guid, :ul_logits.shape[1]], dim=-1)[1][ul_real_span_mask_ltoken.bool()]
            
        ## Update soft-labels
        if self.current_epoch >= self.args.update_soft_start:
            
            score_pad = torch.zeros([score_prot.shape[0], self.num_span_upperb-score_prot.shape[1], score_prot.shape[-1]]).to(self.device)
            score_pad[:,:,0] = 1.0 # default padding to O
            
            if self.args.auto_margin:
                classwise_proto_margin = self.auto_margin
                margin_sum = 999 # flag                
            else:
                classwise_proto_margin = [float(m) for m in self.args.proto_margin.split(',')]
                margin_sum = sum(classwise_proto_margin)
                if len(classwise_proto_margin) == 1:
                    classwise_proto_margin = torch.tensor(classwise_proto_margin * (self.args.n_class - 1)).to(self.device)
                elif len(classwise_proto_margin) == self.args.n_class - 1:
                    classwise_proto_margin = torch.tensor(classwise_proto_margin).to(self.device)
                else:
                    raise Exception("proto_margin has to be either 1 entry or 3 entry, found: ", classwise_proto_margin)

            # class specific update rates
            #update_rate = torch.tensor([float(r) for r in self.args.update_rate.split(',')]).to(self.device)
            update_rate = torch.tensor([1.0 for _ in range(self.args.n_class)]).to(self.device)
            update_rate = update_rate[None, None, :].expand(score_prot.shape[0], score_prot.shape[1], -1)
            update_rate = torch.cat((update_rate, score_pad), dim=1)  
            # freeze spans whose pseudos are already LOC
            # if self.args.update_rate != '1,1,1,1':
            #     pseudo_pred = torch.argmax(self.loss_fn_ul.confidence[unlabel_guid,:score_prot.shape[1]], dim=-1) 
            #     update_rate = torch.where(pseudo_pred.unsqueeze(-1).repeat(1, 1, self.args.n_class) != 3, update_rate, torch.tensor([0.]).to(self.device))

                            
            if margin_sum > 0:
            #if self.args.proto_margin > 0:
                # score_prot_mask = torch.sum(score_prot[:,:,1:] > self.args.proto_margin, dim=-1).bool().unsqueeze(-1).expand(-1, -1, self.args.n_class)
                # Note: torch.expand reuses the same memory, use torch.repeat instead
                ## Option 1: filter before softmax
                # score_prot_mask = torch.sum(logits_prot[:,:,1:] > self.args.proto_margin, dim=-1).bool().unsqueeze(-1).repeat(1, 1, self.args.n_class)
                score_prot_mask = torch.sum(logits_prot[:,:,1:] > classwise_proto_margin[None,None,:], dim=-1).bool().unsqueeze(-1).repeat(1, 1, self.args.n_class)
                ## Option 2: filter after softmax
                # score_prot_mask = torch.sum(score_prot[:,:,1:] > self.args.proto_margin, dim=-1).bool().unsqueeze(-1).repeat(1, 1, self.args.n_class)
                score_prot_mask[:,:,0] = True # Always update spans closest to O cluster
                update_rate = torch.cat((score_prot_mask.long(), score_pad), dim=1) # Do not update out of margin spans
                ## treat out of margin spans as O
                # non_entity_score = torch.zeros_like(score_prot)
                # non_entity_score[:,:,0] = 1.0
                # score_prot = torch.where(score_prot_mask, score_prot, non_entity_score)
                
              
                
            if self.args.postprocess_prot:
                score_prot_idx = torch.max(score_prot, dim=-1)[1] # (bs, n_span)
                _, _, score_prot_idx_new = get_pruning_predIdxs(score_prot_idx, ul_all_span_idxs, score_prot.tolist())
                score_prot_mask = torch.logical_or(score_prot_idx_new.to(torch.device('cuda')) != 0, score_prot_idx == 0).unsqueeze(-1).repeat(1, 1, self.args.n_class)
                update_rate = update_rate * torch.cat((score_prot_mask.long(), score_pad), dim=1)
                    
            score_prot = torch.cat((score_prot, score_pad), dim=1)
            
            self.loss_fn_ul.confidence_update(temp_un_conf=score_prot, batch_index=unlabel_guid, update_rate=update_rate)

        ## Self-training loss     
        loss_ul = self.loss_fn_ul(ul_logits, unlabel_guid, ul_real_span_mask_ltoken.bool(), ul_all_span_weights, ul_all_span_idxs)
        
        ## Contrastive loss
        
        # 2nd pass
        all_span_rep2, q2, _ = self.forward(loadall,all_span_lens,all_span_idxs_ltoken,tokens, attention_mask, token_type_ids, output_q=True)
        ul_all_span_rep2, ul_q2, _ = self.forward(ul_loadall,ul_all_span_lens,ul_all_span_idxs_ltoken,ul_tokens, ul_attention_mask, ul_token_type_ids, output_q=True)
        
        
        # Downsampling non-entities for contrastive loss
        if self.args.cl_downsample < 1.0:
            random_mask = torch.rand(real_span_mask_ltoken.shape).to(self.device) < self.args.cl_downsample
            random_mask = torch.logical_or(random_mask, span_label_ltoken != 0)
            real_span_mask_ltoken = torch.logical_and(random_mask, real_span_mask_ltoken.bool()) #replaced real_span_mask_ltoken varaible
            ul_random_mask = torch.rand(ul_real_span_mask_ltoken.shape).to(self.device) < self.args.cl_downsample
            ul_random_mask = torch.logical_or(ul_random_mask,  pseudo_labels != 0)
            ul_real_span_mask_ltoken = torch.logical_and(ul_random_mask, ul_real_span_mask_ltoken.bool()) #replaced ul_real_span_mask_ltoken varaible
                   
        # Prepare for contrastive loss
        flat_q = q[real_span_mask_ltoken.bool()]
        flat_q2 = q2[real_span_mask_ltoken.bool()]
        flat_ul_q = ul_q[ul_real_span_mask_ltoken.bool()]
        flat_ul_q2 = ul_q2[ul_real_span_mask_ltoken.bool()]
        features = torch.cat((flat_q, flat_ul_q, flat_q2, flat_ul_q2), dim=0)

        flat_labels = torch.nn.functional.one_hot(span_label_ltoken[real_span_mask_ltoken.bool()], num_classes=self.args.n_class).float()
        # Postprocess for contrastive loss
        if self.args.postprocess_cl:
            predicted_scores_idx = torch.max(predicted_scores, dim=-1)[1] # (bs, n_span)
            _, _, predicted_scores_idx_new = get_pruning_predIdxs(predicted_scores_idx, ul_all_span_idxs, predicted_scores.tolist())
            postprocess_mask = torch.logical_or(predicted_scores_idx_new.to(torch.device('cuda')) != 0, predicted_scores_idx == 0)
            non_entity_score = torch.zeros_like(predicted_scores)
            non_entity_score[:,:,0] = 1.0
            predicted_scores = torch.where(postprocess_mask.unsqueeze(-1), predicted_scores, non_entity_score)
            
        flat_predicted_scores = predicted_scores[ul_real_span_mask_ltoken.bool()]
        pseudo_scores = torch.cat((flat_labels, flat_predicted_scores, flat_labels, flat_predicted_scores), dim=0)

        #TODO: queue

        pseudo_target_max, pseudo_target_cont = torch.max(pseudo_scores, dim=1)
        pseudo_target_cont = pseudo_target_cont.contiguous().view(-1, 1) # What does this do???

        # Mask for SupCon loss
        total_batch_size = flat_labels.shape[0] + flat_predicted_scores.shape[0]
        supcon_mask = torch.eq(pseudo_target_cont[:total_batch_size], pseudo_target_cont.T).float().cuda()
        
        # SupCon loss
        loss_cont = self.loss_cont_fn(features=features, mask=supcon_mask, batch_size=total_batch_size)
        
        loss_rdrop = 0.0
        if self.args.rdrop_weight > 0.0:
            flat_label_logits = all_span_rep.view([-1, all_span_rep.shape[-1]])
            flat_label_logits2 = all_span_rep2.view([-1, all_span_rep2.shape[-1]])
            flat_label_pad_mask = ~real_span_mask_ltoken.bool().view([-1]) # 1 where pad token and 0 otherwise
            
#             flat_ul_logits = ul_all_span_rep.view([-1, ul_all_span_rep.shape[-1]])
#             flat_ul_logits2 = ul_all_span_rep2.view([-1, ul_all_span_rep2.shape[-1]])
#             flat_ul_pad_mask = ~ul_real_span_mask_ltoken.bool().view([-1]) # 1 where pad token and 0 otherwise
            
#             total_logits = torch.cat((flat_label_logits, flat_ul_logits), dim=0)
#             total_logits2 = torch.cat((flat_label_logits2, flat_ul_logits2), dim=0)
#             total_pad_mask = torch.cat((flat_label_pad_mask, flat_ul_pad_mask), dim=0)

#             loss_rdrop = compute_kl_loss(total_logits, total_logits2, pad_mask=total_pad_mask)
            
            loss_rdrop = compute_kl_loss(flat_label_logits, flat_label_logits2, pad_mask=flat_label_pad_mask)
       
        

        # Total loss
        
        loss = loss + 1.0 * loss_cont + 1.0 * loss_ul + self.args.rdrop_weight * loss_rdrop
        
        output[f"train_loss"] = loss

        tf_board_logs[f"loss"] = loss

        output['loss'] = loss
        output['log'] =tf_board_logs
        
        
        # Analysis on pseudo labels
        skip_overlap_pseudo = False
        if skip_overlap_pseudo:
            ul_span_f1s,ul_pred_label_idx = span_f1_prune(ul_all_span_idxs, 
                                                          self.loss_fn_ul.confidence[unlabel_guid,:ul_logits.shape[1]], 
                                                          ul_span_label_ltoken, 
                                                          ul_real_span_mask_ltoken)
            ul_batch_preds = get_predict_prune(self.args, ul_all_span_word, ul_words, 
                                               ul_pred_label_idx, ul_span_label_ltoken, ul_all_span_idxs)
        else:
            ul_span_f1s = span_f1(self.loss_fn_ul.confidence[unlabel_guid,:ul_logits.shape[1]], ul_span_label_ltoken, ul_real_span_mask_ltoken)
            ul_batch_preds = get_predict(self.args, ul_all_span_word, ul_words, 
                                         self.loss_fn_ul.confidence[unlabel_guid,:ul_logits.shape[1]], 
                                         ul_span_label_ltoken, ul_all_span_idxs)
        output['ul_span_f1s'] = ul_span_f1s
        output['ul_batch_preds'] = ul_batch_preds   

        # Visualize labeled & unlabeled data distribution
        # output['label_q'] = q[real_span_mask_ltoken.bool()]
        # output['label_gold_label'] = span_label_ltoken[real_span_mask_ltoken.bool()]
        # output['unlabel_q'] = ul_q[ul_real_span_mask_ltoken.bool()]
        # output['unlabel_gold_label'] = ul_span_label_ltoken[ul_real_span_mask_ltoken.bool()]
        output['label_q'] = last_rep[real_span_mask_ltoken.bool()].clone().detach().cpu()
        output['label_gold_label'] = span_label_ltoken[real_span_mask_ltoken.bool()].clone().detach().cpu()
        output['unlabel_q'] = ul_last_rep[ul_real_span_mask_ltoken.bool()].clone().detach().cpu()
        output['unlabel_gold_label'] = ul_span_label_ltoken[ul_real_span_mask_ltoken.bool()].clone().detach().cpu()
        
        return output


    def training_epoch_end(self, outputs):
        """"""
        print("use... training_epoch_end: ", )
        avg_loss = torch.stack([x['train_loss'] for x in outputs]).mean()
        tensorboard_logs = {'train_loss': avg_loss}
        all_counts = torch.stack([x[f'span_f1s'] for x in outputs]).sum(0)
        correct_pred, total_pred, total_golden = all_counts
        print('in train correct_pred, total_pred, total_golden: ', correct_pred, total_pred, total_golden)
        precision =correct_pred / (total_pred+1e-10)
        recall = correct_pred / (total_golden + 1e-10)
        f1 = precision * recall * 2 / (precision + recall + 1e-10)

        print("in train span_precision: ", precision)
        print("in train span_recall: ", recall)
        print("in train span_f1: ", f1)
        tensorboard_logs[f"span_precision"] = precision
        tensorboard_logs[f"span_recall"] = recall
        tensorboard_logs[f"span_f1"] = f1

        self.fwrite_epoch_res.write(
            "train: %f, %f, %f, %d, %d, %d\n" % (f1, recall, precision, correct_pred, total_pred, total_golden))
        
        #begin{analyze pseudo label quality}
        ul_all_counts = torch.stack([x[f'ul_span_f1s'] for x in outputs]).sum(0)
        ul_correct_pred, ul_total_pred, ul_total_golden = ul_all_counts
        print('ul_correct_pred, ul_total_pred, ul_total_golden: ', ul_correct_pred, ul_total_pred, ul_total_golden)
        ul_precision =ul_correct_pred / (ul_total_pred+1e-10)
        ul_recall = ul_correct_pred / (ul_total_golden + 1e-10)
        ul_f1 = ul_precision * ul_recall * 2 / (ul_precision + ul_recall + 1e-10)

        print("ul_span_precision: ", ul_precision)
        print("ul_span_recall: ", ul_recall)
        print("ul_span_f1: ", ul_f1)
        self.fwrite_epoch_res.write("pseudo: %f, %f, %f, %d, %d, %d\n"%(ul_f1,ul_recall,ul_precision,ul_correct_pred, ul_total_pred, ul_total_golden) )
        
        ul_pred_batch_results = [x['ul_batch_preds'] for x in outputs]
        fp_write = self.args.default_root_dir +  '/' + self.args.modelName + '_pseudo' + str(self.current_epoch) + '.txt'
        fwrite = open(fp_write, 'w')
        for ul_pred_batch_result in ul_pred_batch_results:
            for ul_pred_result in ul_pred_batch_result:
                # print("pred_result: ", pred_result)
                fwrite.write(ul_pred_result + '\n')
        #end{analyze pseudo label quality}

        #begin{Visualize label & unlabel distribution}
        label_q = torch.cat([x[f'label_q'] for x in outputs], dim=0).cpu().detach().numpy()
        label_gold_label = torch.cat([x[f'label_gold_label'] for x in outputs], dim=0).cpu().detach().numpy()
        unlabel_q = torch.cat([x[f'unlabel_q'] for x in outputs], dim=0).cpu().detach().numpy()
        unlabel_gold_label = torch.cat([x[f'unlabel_gold_label'] for x in outputs], dim=0).cpu().detach().numpy()
        with open(self.args.default_root_dir +  '/' + 'label_q_ep' + str(self.current_epoch) + '.npy', 'wb') as f:
            np.save(f, label_q)
        with open(self.args.default_root_dir +  '/' + 'label_gold_label_ep' + str(self.current_epoch) + '.npy', 'wb') as f:
            np.save(f, label_gold_label)
        with open(self.args.default_root_dir +  '/' + 'unlabel_q_ep' + str(self.current_epoch) + '.npy', 'wb') as f:
            np.save(f, unlabel_q)
        with open(self.args.default_root_dir +  '/' + 'unlabel_gold_label_ep' + str(self.current_epoch) + '.npy', 'wb') as f:
            np.save(f, unlabel_gold_label)
        
        

        if self.current_epoch >= self.args.update_soft_start:
            epoch_logits_prot = torch.cat([x[f'logits_prot'] for x in outputs], dim=0).cpu().detach().numpy()
            epoch_score_prot = torch.cat([x[f'score_prot'] for x in outputs], dim=0).cpu().detach().numpy()
            unlabel_pred_label = torch.cat([x[f'unlabel_pred_label'] for x in outputs], dim=0).cpu().detach().numpy()
            
            with open(self.args.default_root_dir +  '/' + 'logits_prot_ep' + str(self.current_epoch) + '.npy', 'wb') as f:
                np.save(f, epoch_logits_prot)
            with open(self.args.default_root_dir +  '/' + 'score_prot_ep' + str(self.current_epoch) + '.npy', 'wb') as f:
                np.save(f, epoch_score_prot)
            with open(self.args.default_root_dir +  '/' + 'unlabel_pred_label_ep' + str(self.current_epoch) + '.npy', 'wb') as f:
                np.save(f, unlabel_pred_label)
                
            print('Radius mean and variance...')    
            total_sim = [[] for _ in range(self.args.n_class)]
            for i in range(epoch_logits_prot.shape[0]):
                label = unlabel_pred_label[i]
                total_sim[label].append(epoch_logits_prot[i])
            auto_margin = []
            for j in range(self.args.n_class):
                cls_sim = np.stack(total_sim[j], axis=0)
                print(np.mean(cls_sim, axis=0)[j], np.std(cls_sim, axis=0)[j])
                if j > 0: # exclude O
                    auto_margin.append(np.mean(cls_sim, axis=0)[j])
            self.auto_margin = torch.FloatTensor(auto_margin).to(self.device)
            
                
            

        #end{Visualize label & unlabel distribution}
        
        
        
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        """"""

        output = {}

        # tokens, token_type_ids, start_labels, end_labels, start_label_mask, end_label_mask, match_labels, sample_idx, label_idx = batch
        tokens, token_type_ids, all_span_idxs_ltoken,morph_idxs, span_label_ltoken, all_span_lens,all_span_weights,real_span_mask_ltoken,words,all_span_word,all_span_idxs = batch
        loadall = [tokens, token_type_ids, all_span_idxs_ltoken,morph_idxs, span_label_ltoken, all_span_lens,all_span_weights,real_span_mask_ltoken,words,all_span_word,all_span_idxs]

        attention_mask = (tokens != 0).long()
        all_span_rep = self.forward(loadall,all_span_lens,all_span_idxs_ltoken, tokens, attention_mask, token_type_ids)
        predicts = self.classifier(all_span_rep)

        # pred_label_idx_new = torch.zeros_like(real_span_mask_ltoken)
        if self.args.use_prune:
            span_f1s,pred_label_idx = span_f1_prune(all_span_idxs, predicts, span_label_ltoken, real_span_mask_ltoken)
            # print('pred_label_idx_new: ',pred_label_idx_new.shape)
            # print('predicts: ', predicts.shape)
            # print('pred_label_idx_new: ',pred_label_idx_new)
            # print('predicts: ', predicts)

            batch_preds = get_predict_prune(self.args, all_span_word, words, pred_label_idx, span_label_ltoken,
                                               all_span_idxs)
        else:
            span_f1s = span_f1(predicts, span_label_ltoken, real_span_mask_ltoken)
            batch_preds = get_predict(self.args, all_span_word, words, predicts, span_label_ltoken,
                                               all_span_idxs)

        output["span_f1s"] = span_f1s
        loss = self.compute_loss(loadall,all_span_rep, span_label_ltoken, real_span_mask_ltoken,mode='test/dev')


        output["batch_preds"] =batch_preds
        # output["batch_preds_prune"] = pred_label_idx_new
        output[f"val_loss"] = loss

        output["predicts"] = predicts
        output['all_span_word'] = all_span_word

        return output

    def validation_epoch_end(self, outputs):
        """"""
        print("use... validation_epoch_end: ", )
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        all_counts = torch.stack([x[f'span_f1s'] for x in outputs]).sum(0)
        correct_pred, total_pred, total_golden = all_counts
        print('correct_pred, total_pred, total_golden: ', correct_pred, total_pred, total_golden)
        precision =correct_pred / (total_pred+1e-10)
        recall = correct_pred / (total_golden + 1e-10)
        f1 = precision * recall * 2 / (precision + recall + 1e-10)

        print("span_precision: ", precision)
        print("span_recall: ", recall)
        print("span_f1: ", f1)
        tensorboard_logs[f"span_precision"] = precision
        tensorboard_logs[f"span_recall"] = recall
        tensorboard_logs[f"span_f1"] = f1
        self.fwrite_epoch_res.write("dev: %f, %f, %f, %d, %d, %d\n"%(f1,recall,precision,correct_pred, total_pred, total_golden) )

        if f1>self.args.best_dev_f1:
            pred_batch_results = [x['batch_preds'] for x in outputs]
            fp_write = self.args.default_root_dir +  '/' + self.args.modelName + '_dev.txt'
            fwrite = open(fp_write, 'w')
            for pred_batch_result in pred_batch_results:
                for pred_result in pred_batch_result:
                    # print("pred_result: ", pred_result)
                    fwrite.write(pred_result + '\n')
            self.args.best_dev_f1=f1

            # begin{save the predict prob}
            all_predicts = [list(x['predicts']) for x in outputs]
            all_span_words = [list(x['all_span_word']) for x in outputs]

            # begin{get the label2idx dictionary}
            label2idx = {}
            label2idx_list = self.args.label2idx_list
            for labidx in label2idx_list:
                lab, idx = labidx
                label2idx[lab] = int(idx)
                # end{get the label2idx dictionary}

            file_prob1 = self.args.default_root_dir + '/' + self.args.modelName + '_prob_dev.pkl'
            print("the file path of probs: ", file_prob1)
            fwrite_prob = open(file_prob1, 'wb')
            pickle.dump([label2idx, all_predicts, all_span_words], fwrite_prob)
            # end{save the predict prob...}


        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        """"""
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(
        self,
        outputs
    ) -> Dict[str, Dict[str, Tensor]]:
        """"""
        print("use... test_epoch_end: ",)
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        all_counts = torch.stack([x[f'span_f1s'] for x in outputs]).sum(0)
        correct_pred, total_pred, total_golden = all_counts
        print('correct_pred, total_pred, total_golden: ', correct_pred, total_pred, total_golden)
        precision = correct_pred / (total_pred + 1e-10)
        recall = correct_pred / (total_golden + 1e-10)
        f1 = precision * recall * 2 / (precision + recall + 1e-10)

        print("span_precision: ", precision)
        print("span_recall: ", recall)
        print("span_f1: ", f1)
        tensorboard_logs[f"span_precision"] = precision
        tensorboard_logs[f"span_recall"] = recall
        tensorboard_logs[f"span_f1"] = f1


        # begin{save the predict results}
        pred_batch_results = [x['batch_preds'] for x in outputs]
        fp_write = self.args.default_root_dir + '/'+self.args.modelName +'_test.txt'
        fwrite = open(fp_write, 'w')
        for pred_batch_result in pred_batch_results:
            for pred_result in pred_batch_result:
                # print("pred_result: ", pred_result)
                fwrite.write(pred_result+'\n')

        self.fwrite_epoch_res.write(
            "test: %f, %f, %f, %d, %d, %d\n" % (f1, recall, precision, correct_pred, total_pred, total_golden))
        # end{save the predict results}


        # begin{save the predict prob}
        all_predicts = [list(x['predicts'].cpu()) for x in outputs]
        all_span_words = [list(x['all_span_word']) for x in outputs]

            # begin{get the label2idx dictionary}
        label2idx = {}
        label2idx_list = self.args.label2idx_list
        for labidx in label2idx_list:
            lab, idx = labidx
            label2idx[lab] = int(idx)
            # end{get the label2idx dictionary}

        file_prob1 = self.args.default_root_dir + '/'+self.args.modelName +'_prob_test.pkl'
        print("the file path of probs: ", file_prob1)
        fwrite_prob = open(file_prob1, 'wb')
        pickle.dump([label2idx, all_predicts, all_span_words], fwrite_prob)
        # end{save the predict prob...}

        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    
    def prepare_data(self):
        # Create repeated loader for unlabeled data
        self.unlabel_loader = self.get_dataloader("unlabel")
        self.unlabel_loader = repeat_dataloader(self.unlabel_loader)
        
    def train_dataloader(self) -> DataLoader:
#         label_loader = self.get_dataloader("train")
#         unlabel_loader = self.get_dataloader("unlabel")
        
#         return {'labeled': label_loader, 'unlabeled': unlabel_loader}
        
        return self.get_dataloader("train")
        # return self.get_dataloader("dev", 100)

    def val_dataloader(self):
        val_data = self.get_dataloader("dev")
        return val_data

    def test_dataloader(self):
        return self.get_dataloader("test")
        # return self.get_dataloader("dev")

    def get_dataloader(self, prefix="train", limit: int = None) -> DataLoader:
        """get training dataloader"""
        """
        load_mmap_dataset
        """
        json_path = os.path.join(self.data_dir, f"spanner.{prefix}")
        print("json_path: ", json_path)
        # vocab_path = os.path.join(self.bert_dir, "vocab.txt")
        # dataset = BERTNERDataset(self.args,json_path=json_path,
        #                         tokenizer=BertWordPieceTokenizer(vocab_path),
        #                         # tokenizer=BertWordPieceTokenizer(vocab_file=vocab_path),
        #                         max_length=self.args.bert_max_length,
        #                         pad_to_maxlen=False
        #                         )

        # vocab_path = os.path.join(self.bert_dir, "vocab.txt")
        # print("use BertWordPieceTokenizer as the tokenizer ")
        # dataset = BERTNERDataset(self.args, json_path=json_path,
        #                          tokenizer=BertWordPieceTokenizer(vocab_path),
        #                          # tokenizer=BertWordPieceTokenizer(vocab_file=vocab_path),
        #                          max_length=self.args.bert_max_length,
        #                          pad_to_maxlen=False
        #                          )
        
        print("use Tokenizer from pretrained XLMR")
        return_guid = True if prefix=='unlabel' else False
        dataset = BERTNERDataset(self.args, json_path=json_path,
                                 tokenizer=Tokenizer.from_pretrained(self.args.bert_config_dir),
                                 max_length=self.args.bert_max_length,
                                 pad_to_maxlen=False,
                                 return_guid=return_guid
                                 )


        if limit is not None:
            dataset = TruncateDataset(dataset, limit)

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.args.batch_size,
            # num_workers=self.args.workers,
            shuffle=True if (prefix == "train" or prefix == "unlabel") else False,
            # shuffle=False,
            drop_last=False,
            collate_fn=collate_to_max_length
        )
        return dataloader
    
# Repeated dataloader
def repeat_dataloader(iterable):
    """ repeat dataloader """
    while True:
        for x in iterable:
            yield x

class partial_loss(torch.nn.Module):
    def __init__(self, confidence, conf_ema_m=0.99, postprocess_pseudo=False):
        super().__init__()
        self.confidence = confidence.to(torch.device('cuda'))
        self.init_conf = confidence.detach()
        self.conf_ema_m = conf_ema_m
        self.postprocess_pseudo = postprocess_pseudo
        if self.postprocess_pseudo:
            print("Will postprocess pseudo labels by filtering overlapping spans...")

    def set_conf_ema_m(self, epoch, args):
        start = args.conf_ema_range[0]
        end = args.conf_ema_range[1]
        self.conf_ema_m = 1. * (epoch - args.update_soft_start) / (args.max_epochs - args.update_soft_start) * (end - start) + start

    def forward(self, outputs, index, pad_mask, weights, all_span_idxs=None, verbose=False):
        postprocess_mask = torch.ones_like(pad_mask, dtype=bool)
        if self.postprocess_pseudo:
            pseudos = self.confidence[index, :outputs.shape[1]]
            pseudo_label_idx = torch.max(pseudos, dim=-1)[1] # (bs, n_span)
            span_probs = pseudos.tolist()
            _, _, pseudo_label_idx_new = get_pruning_predIdxs(pseudo_label_idx, all_span_idxs, span_probs)
            postprocess_mask = torch.logical_or(pseudo_label_idx_new.to(torch.device('cuda')) != 0, pseudo_label_idx == 0)

            filtered_span_mask = torch.logical_and(pseudo_label_idx!=0, pseudo_label_idx_new.to(torch.device('cuda')) == 0)
            filtered_logsm_outputs = F.log_softmax(outputs[filtered_span_mask * pad_mask], dim=1)
            filtered_outputs = filtered_logsm_outputs[:,0] # treat as one-hot O spans

        logsm_outputs = F.log_softmax(outputs[pad_mask * postprocess_mask], dim=1)
        final_outputs = logsm_outputs * self.confidence[index, :outputs.shape[1]][pad_mask * postprocess_mask]

        if self.postprocess_pseudo and filtered_outputs.nelement() > 0:
            average_loss = - torch.cat((final_outputs.sum(dim=1) * weights[pad_mask * postprocess_mask], filtered_outputs), dim=0).mean()
        else:
            average_loss = - (final_outputs.sum(dim=1) * weights[pad_mask * postprocess_mask]).mean()
        
        # logsm_outputs = F.log_softmax(outputs[pad_mask], dim=1)
        # final_outputs = logsm_outputs * self.confidence[index, :][pad_mask]
        # average_loss = - (final_outputs.sum(dim=1) * weights[pad_mask]).mean()
        
        return average_loss

    def confidence_update(self, temp_un_conf, batch_index, update_rate):
        with torch.no_grad():
            _, prot_pred = temp_un_conf.max(dim=-1)
            pseudo_label = F.one_hot(prot_pred, temp_un_conf.shape[-1]).float().cuda().detach()
            # pseudo update rate
            pseudo_label = pseudo_label * update_rate
            self.confidence[batch_index, :] = self.conf_ema_m * self.confidence[batch_index, :]\
                 + (1 - self.conf_ema_m) * pseudo_label
            self.confidence[batch_index, :] = F.normalize(self.confidence[batch_index, :], p=1.0, dim=-1) # normalize to sum to 1
            
        return None    

class SupConLoss(torch.nn.Module):
    """Following Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf."""
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, mask=None, batch_size=-1):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if mask is not None:
            # SupCon loss (Partial Label Mode)
            mask = mask.float().detach().to(device)
            # compute logits
            anchor_dot_contrast = torch.div(
                torch.matmul(features[:batch_size], features.T),
                self.temperature)
            # for numerical stability
            logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
            logits = anchor_dot_contrast - logits_max.detach()
 
            # mask-out self-contrast cases
            logits_mask = torch.scatter(
                torch.ones_like(mask),
                1,
                torch.arange(batch_size).view(-1, 1).to(device),
                0
            )
            mask = mask * logits_mask

            # compute log_prob
            exp_logits = torch.exp(logits) * logits_mask
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

            # compute mean of log-likelihood over positive
            mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

            # loss
            loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
            loss = loss.mean()
        else:
            # MoCo loss (unsupervised)
            # compute logits
            # Einstein sum is more intuitive
            # positive logits: Nx1
            q = features[:batch_size]
            k = features[batch_size:batch_size*2]
            queue = features[batch_size*2:]
            l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
            # negative logits: NxK
            l_neg = torch.einsum('nc,kc->nk', [q, queue])
            # logits: Nx(1+K)
            logits = torch.cat([l_pos, l_neg], dim=1)

            # apply temperature
            logits /= self.temperature

            # labels: positive key indicators
            labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
            loss = F.cross_entropy(logits, labels)

        return loss    

def compute_kl_loss(p, q, pad_mask=None):
    
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')
    
    p_loss = p_loss.sum(dim=-1)
    q_loss = q_loss.sum(dim=-1)
    
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

    p_loss = p_loss.sum() / (~pad_mask).sum() #mean
    q_loss = q_loss.sum() / (~pad_mask).sum() #mean

    loss = (p_loss + q_loss) / 2
    return loss     
    
def main():
    """main"""
    # parser = get_parser()

    # add model specific args
    parser = BertNerTagger.get_parser()

    # add all the available trainer options to argparse
    # ie: now --gpus --num_nodes ... --fast_dev_run all work in the cli
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()
    args.conf_ema_range = [float(item) for item in args.conf_ema_range.split(',')]
    
    set_random_seed(args.seed)

    # begin{add label2indx augument into the args.}
    label2idx = {}
    if 'conll' in args.dataname:
        label2idx = {"O": 0, "ORG": 1, "PER": 2, "LOC": 3, "MISC": 4}
    elif 'note' in args.dataname:
        label2idx = {'O': 0, 'PERSON': 1, 'ORG': 2, 'GPE': 3, 'DATE': 4, 'NORP': 5, 'CARDINAL': 6, 'TIME': 7,
                     'LOC': 8,
                     'FAC': 9, 'PRODUCT': 10, 'WORK_OF_ART': 11, 'MONEY': 12, 'ORDINAL': 13, 'QUANTITY': 14,
                     'EVENT': 15,
                     'PERCENT': 16, 'LAW': 17, 'LANGUAGE': 18}
    elif args.dataname == 'wnut16':
        label2idx = {'O': 0, 'loc':1, 'facility':2,'movie':3,'company':4,'product':5,'person':6,'other':7,
                     'tvshow':8,'musicartist':9,'sportsteam':10}
    elif args.dataname == 'wnut17':
        label2idx = {'O': 0,'location':1, 'group':2,'corporation':3,'person':4,'creative-work':5,'product':6}
    elif 'wikiann' in args.dataname:
        label2idx = {"O": 0, "ORG": 1, "PER": 2, "LOC": 3}

    label2idx_list = []
    for lab, idx in label2idx.items():
        pair = (lab, idx)
        label2idx_list.append(pair)
    args.label2idx_list = label2idx_list
    # end{add label2indx augument into the args.}

    # begin{add case2idx augument into the args.}
    morph2idx_list = []
    morph2idx = {'isupper': 1, 'islower': 2, 'istitle': 3, 'isdigit': 4, 'other': 5}
    for morph, idx in morph2idx.items():
        pair = (morph, idx)
        morph2idx_list.append(pair)
    args.morph2idx_list = morph2idx_list
    # end{add case2idx augument into the args.}

    #args.default_root_dir = args.default_root_dir+'_'+args.random_int
    args.default_root_dir = args.default_root_dir+'/run'+str(args.seed)

    if not os.path.exists(args.default_root_dir):
        os.makedirs(args.default_root_dir)

    fp_epoch_result = args.default_root_dir+'/epoch_results.txt'
    args.fp_epoch_result =fp_epoch_result




    text = '\n'.join([hp for hp in str(args).replace('Namespace(', '').replace(')', '').split(', ')])
    print(text)

    text = '\n'.join([hp for hp in str(args).replace('Namespace(', '').replace(')', '').split(', ')])
    fn_path = args.default_root_dir + '/' +args.param_name+'.txt'
    if fn_path is not None:
        with open(fn_path, mode='w') as text_file:
            text_file.write(text)

    model = BertNerTagger(args)
    if args.pretrained_checkpoint:
        model.load_state_dict(torch.load(args.pretrained_checkpoint,
                                         map_location=torch.device('cpu'))["state_dict"])

    # save the best model
    checkpoint_callback = ModelCheckpoint(
        filepath=args.default_root_dir,
        save_top_k=1,
        verbose=True,
        monitor="span_f1",
        period=-1,
        mode="max",
    )
    trainer = Trainer.from_argparse_args(
        args,
        checkpoint_callback=checkpoint_callback
    )

    trainer.fit(model)
    trainer.test()


if __name__ == '__main__':
    # run_dataloader()
    main()
