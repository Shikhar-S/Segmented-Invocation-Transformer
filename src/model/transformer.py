from numpy import inf, mat
import torch
import torch.nn as nn
import pytorch_lightning as pl
from argparse import ArgumentParser
from main_utils import str2bool, get_logger
from model.loss_functions import LabelSmoothingKLLoss
from decoder.bash_generator import get_score
from decoder.translator import TranslationBuilder
logger = get_logger()

class Transformer(pl.LightningModule):
    def __init__(self, encoder, decoder, d_model, trg_tokenizer, device):
        super(Transformer, self).__init__()
        self.translator = None
        self.invocation_encoder = encoder
        self.decoder = decoder
        self.d_model = d_model
        self.pad = trg_tokenizer.pad
        self.trg_tokenizer = trg_tokenizer
        self.inference = False
        self.generator_head = nn.Linear(d_model,trg_tokenizer.vocab_len)

    def init_translator(self,translator):
        self.translator = translator

    def forward(self, sentence_phrases, phrases_len, target, target_len):
        '''
        sentence_phrases [batch size X max phrase count X max phrase length]
        phrases_len  [batch size X max phrase count]
        target [max target length X batch X 1]
        '''
        encodings, lengths = self.invocation_encoder(sentence_phrases, phrases_len)
        # out [max phrase count X batch size X model dim]
        # lengths [batch size]
        #################################INPUT FOR DECODER################################
        dec_out = self.decoder(target, memory_bank = encodings, memory_lengths = lengths)

        dist_out = self.generator_head(dec_out)
        #T B C

        logsoftmax = nn.LogSoftmax(dim=-1)
        log_dis = logsoftmax(dist_out)

        return log_dis


    def _cross_attn_mask(self,k_lens,key_max_len):
        batch_size = k_lens.shape[0]
        mask_v = (torch.arange(0, key_max_len, device=k_lens.device)
                .type_as(k_lens)
                .repeat(batch_size, 1)
                .lt(k_lens.unsqueeze(1))).unsqueeze(1)
        return ~mask_v

    def update_dropout(self, dropout):
        self.invocation_encoder.update_dropout(dropout)
        self.value_decoder.update_dropout(dropout)

    ################################# LIGHTNING FUNCTIONS ###################################################
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())


    def shared_step(self,batch,batch_idx,compute_metric = False):
        (phrase_batch, phrase_len_batch),(y,y_len)= batch
        y = y.transpose(1,0)

        output = self(sentence_phrases = phrase_batch,\
            phrases_len = phrase_len_batch,\
            target = y,\
            target_len = y_len)
        # length X batch X classes

        #compute loss
        # loss_fn = nn.NLLLoss(ignore_index = self.pad)
        loss_fn = LabelSmoothingKLLoss(label_smoothing=0.1,tgt_vocab_size=self.trg_tokenizer.vocab_len,ignore_index=self.pad)
        loss = loss_fn(output[:-1,:,:].permute(1,2,0),y[1:,:,0].transpose(1,0))
        loss = loss / y_len.sum();

        #GOLD METRIC
        # if compute_metric:
        #     metric_score = self.compute_gold_metric(output,y,y_len,leaves_batch)
        # else:
        #     metric_score = -1
        #BEAM SEARCH METRIC
        if compute_metric:
            metric_score = self.compute_metric_score(phrase_batch, phrase_len_batch, y,y_len)
        else:
            metric_score = -1

        return loss, metric_score

    def training_step(self, batch, batch_idx):
        compute_metric = False
        loss, metric_score = self.shared_step(batch,batch_idx,compute_metric=compute_metric)
        self.log('loss/training',loss)
        if compute_metric:
            self.log('metric/training',metric_score)
        return loss

    def validation_step(self, batch, batch_idx):
        compute_metric = True
        loss, metric_score = self.shared_step(batch,batch_idx,compute_metric=compute_metric)
        self.log('loss/validation',loss,sync_dist=True)
        if compute_metric:
            self.log('metric/validation',metric_score,sync_dist=True)
        return loss

    def compute_gold_metric(self,y_hat,y,y_len,inv):
        #y_hat : len , batch, dic
        #y: len, batch, 1
        y_hat = torch.argmax(y_hat,dim=-1)
        y_hat = y_hat.transpose(1,0)
        inv_len = (inv!=self.pad).sum(dim=1)

        translation_builder = TranslationBuilder(self.translator._src_vocab.leaf,self.translator._tgt_vocab)
        translations = translation_builder.build_golden_batch(y_hat,\
                y,y_len, inv,\
                inv_len)
        truth = [[" ".join(t) for t in translation.truth] for translation in translations]
        pred = [[" ".join(p) for p in translation.pred] for translation in translations]
        inv_text = [" ".join(translation.inv) for translation in translations]
        scores,_ = get_score(truth, pred)
        return scores.mean()

    def compute_metric_score(self, phrase_batch, phrase_len_batch, y,y_len):
        if self.translator is None:
            return -1
        self.inference = True
        with torch.no_grad():
            translations = self.translator.translate(phrase_batch, phrase_len_batch, y,y_len)
            truth = [[" ".join(t) for t in translation.truth] for translation in translations]
            pred = [[" ".join(p) for p in translation.pred] for translation in translations]
            inv_text = [" ".join(translation.inv) for translation in translations]
            # print(inv_text)
            # print(truth)
            # print(pred)
            # print('--'*40)
            scores,_ = get_score(truth, pred)
        mean_score = scores.mean()
        self.inference=False

        return mean_score

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--log_hist",type=str2bool,default=False)
        parser.add_argument('--d_ff',type = int, default = 1024)
        parser.add_argument('--d_model',type = int, default = 256)
        parser.add_argument('--encoder_layers',type = int, default = 3)
        parser.add_argument('--encoder_heads',type = int,default = 4)
        parser.add_argument('--decoder_layers',type = int, default = 6)
        parser.add_argument('--decoder_heads',type = int,default = 8)
        parser.add_argument('--attention_dropout',type = float,default = 0.5377)
        parser.add_argument('--dropout',type = float,default = 0)
        parser.add_argument('--length_penalty',type=float,default=0.4565)
        parser.add_argument('--decoding_strategy',type=str,default = 'beam')
        parser.add_argument('--accumulate_grad_batches',type = int, default = 150)
        parser.add_argument('--beam_size',type = int, default = 10)
        parser.add_argument('--n_best',type=int,default=5)
        parser.add_argument('--embedding_max_norm',type=float,default=9.566)
        return parser
