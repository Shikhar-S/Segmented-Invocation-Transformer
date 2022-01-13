import pickle
from main_utils import check_path
from reader.datamodule import DataModule
import torch
# from captum.attr import LayerIntegratedGradients
import os
from utils.metric_utils import compute_metric

class Attributor:
    def __init__(self,args):
        print('Setting up data module')
        self.dm = DataModule(args)
        self.dm.setup()
        print('subtree cut height=',self.dm.subtree_cut_height)
        self.attribution_records = []

    def init_attr(self,model):
        print('Setting up model')
        self.model = model
        self.lig = LayerIntegratedGradients(forward_func = self.attr_forward_,\
                                            layer = model.invocation_encoder.embedding,\
                                            multiply_by_inputs = True)
        #.pe for positional

    def build_attribution_inputs_(self, batch, time_idx=0):
        '''
        Returns
            baselines :  pad index
            inputs : [batch_cnt, max_phrase_cnt, max_phrase_len, 1]
            additional_args : tuple of (phrase_len, target, target_len) with each tensor same size as in input.
            target : A list of tuples with length equal to the number of examples in inputs (dim 0),
            and each tuple containing #output_dims - 1 elements. Each tuple is applied as the target for the corresponding example.
        '''
        (phrase_batch, phrase_len_batch),(y,y_len)= batch
        baselines = self.dm.src_tokenizer.pad
        additional_args = (phrase_len_batch, y, y_len)
        batch_size = phrase_len_batch.shape[0]
        target = [(time_idx,y[b,time_idx,0].detach().cpu().item()) for b in range(batch_size)]
        return baselines, (phrase_batch), additional_args, target

    def attr_forward_(self,sentence_phrases, phrases_len, target, target_len):
        # attribute on sentence_phrases only
        target = target.transpose(1,0)
        output = self.model(sentence_phrases, phrases_len, target, target_len) # length X batch X classes
        tgt_wrd_prob = output.transpose(0,1)
        return tgt_wrd_prob

    def translate_with_model_(self,batch):
        (phrase_batch, phrase_len_batch),(y,y_len)= batch
        y = y.transpose(1,0)
        with torch.no_grad():
            # print('---'*20)
            # print(phrase_len_batch.device,phrase_batch.device,y.device, y_len.device)
            # print(self.model.device)
            # print('---'*20)
            translations = self.model.translator.translate(phrase_batch, phrase_len_batch, y,y_len)
        return translations

    def get_max_target_length(self, batch):
        (phrase_batch, phrase_len_batch),(y,y_len)= batch
        return y.shape[1]

    def worth_attribution(self, translation, threshold):
        cmd = ' '.join(translation[0].truth[0])
        pred = ' '.join(translation[0].pred[0])
        score = compute_metric(cmd, 1, pred, {'u1':1.0,'u2':1.0})
        if score >= threshold:
            print('Truth',cmd)
            print('Pred',pred)
            return True
        else:
            return False

    def run_integrated_gradients(self, dataloader_type='test'):
        # Get input from dataloader
        if dataloader_type=='test':
            dl = self.dm.test_dataloader()
        elif dataloader_type=='val':
            dl = self.dm.val_dataloader()
        else:
            dl = self.dm.train_dataloader()

        idx = 0
        for batch in dl:
            idx+=1
            # Run input through translator for prediction
            translations = self.translate_with_model_(batch)
            max_target_len = self.get_max_target_length(batch)
            # Construct Baseline for input
            # print(translations[0].inv_phrasing)
            # print(translations[0].truth[0])
            attribute_flag = self.worth_attribution(translations, threshold=1.0)
            if not attribute_flag:
                continue
            inner_attr_records = []
            for time_idx in range(1,max_target_len):
                self.model.zero_grad()
                baselines, inputs, additional_args, target = self.build_attribution_inputs_(batch, time_idx)
                # Integrate Gradients
                attr, err = self.lig.attribute(inputs=inputs,\
                                                baselines=baselines,\
                                                additional_forward_args=additional_args,\
                                                target=target,\
                                                return_convergence_delta=True,n_steps=5000,\
                                                internal_batch_size = 500)
                attr = attr.sum(-1)
                # print(attr[0]/torch.norm(attr[0]))
                attr_record = attr, err
                # Add to attribution record
                inner_attr_records.append(attr_record)
            if idx%1 == 0:
                print(idx)
                print('=='*30)
            self.attribution_records.append((inner_attr_records,translations))
            with open(f'../../run/attribution/test_attr/test5k_attr{idx}.pkl','wb') as g:
                pickle.dump(self.attribution_records[-1],g)


    def save_results(self, filepath):
        import pickle
        check_path(filepath,exist_ok=True)
        with open(filepath,'wb') as f:
            pickle.dump(self.attribution_records,f)