from numpy.core.fromnumeric import shape
from numpy.lib import pad
from main_utils import get_logger, pad_all_, timeit
logger = get_logger()

import torch
from torch.utils.data import DataLoader
from reader.vocab import Vocab
from reader.datareader import DataReader
from reader.sampler import DynamicSampler, StaticSampler
import pickle
from main_utils import str2bool,get_logger, check_path
from argparse import ArgumentParser
import numpy as np
import os
import pytorch_lightning as pl
import pathlib
import nltk
logger = get_logger()

            

class DataModule(pl.LightningDataModule):
    def __init__(self, args):
        super(DataModule,self).__init__()
        self.sampler = args.sampler
        self.multigpu = args.multigpu
        self.is_set = False
        self.process_data = args.process_data
        self.batch_size = args.batch
        data_base_address = args.data_base_address
        self.split = args.split
        self.n_training_examples = args.n_training_examples
        self.subtree_cut_height = args.subtree_cut_height
        cmd_extension = '.cmd.raw' 
        eng_extension = '.en.processed'
        tree_extension = '.tree.pickle'
        dyn_path = pathlib.Path(__file__).parent.parent.parent.absolute()
        self.training_data_path = os.path.join(dyn_path,data_base_address,'train.' +str(self.split)+ eng_extension),\
                                os.path.join(dyn_path,data_base_address,'train.' +str(self.split) + cmd_extension),\
                                os.path.join(dyn_path,data_base_address,'trees/train.' +str(self.split) + tree_extension)
        self.validation_data_path = os.path.join(dyn_path,data_base_address,'valid.' +str(self.split) + eng_extension),\
                                os.path.join(dyn_path,data_base_address,'valid.' +str(self.split) + cmd_extension),\
                                os.path.join(dyn_path,data_base_address,'trees/valid.' +str(self.split) + tree_extension)
        self.testing_data_path = os.path.join(dyn_path,data_base_address,'test.' +str(self.split) + eng_extension),\
                                os.path.join(dyn_path,data_base_address,'test.' +str(self.split)+ cmd_extension),\
                                os.path.join(dyn_path,data_base_address,'trees/test.' +str(self.split) + tree_extension)

        self.extra_data_paths = [(os.path.join(dyn_path,data_base_address,datasource + eng_extension),\
                                os.path.join(dyn_path,data_base_address,datasource + cmd_extension),\
                                os.path.join(dyn_path,data_base_address,datasource + tree_extension))
                                for datasource in args.extra_data_path]

        run_base_address = args.run_base_address
        self.run_address = os.path.join(dyn_path,run_base_address,'split.' + str(self.split)+ '.' + str(self.n_training_examples))
        self.src_tokenizer = None
        self.trg_tokenizer = None
        
        if args.embedding_path != 'None':
            self.embedding_path = os.path.join(dyn_path,args.embedding_path)
        else:
            self.embedding_path = None
        self.device = args.device
        self.documentation_path = os.path.join(dyn_path,args.documentation_path)


    def get_data(self,path,dtype = 'train'):
        datapath = os.path.join(self.run_address, dtype + f'_transformer_pie_conv_{self.subtree_cut_height}.pkl')
        check_path(datapath,exist_ok = True)
        logger.info(f'DataPath: {datapath}')
        print(f'DataPath: {datapath}')
        if True or self.process_data or not os.path.exists(datapath):
            logger.info(f'Processing {dtype} data')

            if dtype == 'train':
                dataset = DataReader(paths = path,\
                    subtree_cut_height = self.subtree_cut_height,\
                    src_tokenizer = self.src_tokenizer,\
                    trg_tokenizer = self.trg_tokenizer,\
                    extra_data_paths = self.extra_data_paths,\
                    n_examples = self.n_training_examples)
            else:
                dataset = DataReader(paths = path,\
                    subtree_cut_height = self.subtree_cut_height,\
                    src_tokenizer = self.src_tokenizer,\
                    trg_tokenizer = self.trg_tokenizer)

            # logger.info(f'Saving {dtype} data')
            # with open(datapath,'wb') as f:
            #     pickle.dump(dataset,f)
            # logger.info(f'Saved {dtype} data')
        else:
            #load from pickled file
            logger.info(f'Loading {dtype} data from pickled file')
            with open(datapath,'rb') as f:
                dataset = pickle.load(f)
        self.src_tokenizer = dataset.src_tokenizer
        self.trg_tokenizer = dataset.trg_tokenizer
        return dataset

    def setup(self, stage=None):
        if not self.is_set:
            self.setup_()
            self.is_set = True

    @timeit
    def setup_(self, stage=None):
        # load training data, save dictionary, load validation and testing data
        self.training_data = self.get_data(self.training_data_path,dtype = 'train')
        self.validation_data = self.get_data(self.validation_data_path,dtype = 'valid')
        self.testing_data = self.get_data(self.testing_data_path,dtype ='test')
        self.filter_tokens(self.src_tokenizer,2500,self.embedding_path)
        self.word_vectors = self.load_word_vec()

    def _get_embedding_words(self,path=None):
        if path is None:
            return []
        embedding_words = set()
        with open(path,'r',encoding='Latin-1') as vec_f:
            n,vec_dim = map(int,vec_f.readline().rstrip().split())
            for line in vec_f:
                word = line.strip().split()[0]
                embedding_words.add(word)
        logger.info(f'Loaded {len(embedding_words)} embedding words')
        return embedding_words


    def filter_tokens(self,full_vocab,k,path=None):
        logger.info('Filtering words from dataset.')
        logger.info(f'Original dictionary size {full_vocab.vocab_len}')
        embedding_words = self._get_embedding_words(path)
        token_counter = full_vocab.token_counter
        top_k_words = set(map(lambda z: z[0],token_counter.most_common(k)))
        filtered_embeddding_words = set()
        for word in embedding_words:
            if word in token_counter.keys():
                filtered_embeddding_words.add(word)
        filtered_words = list(top_k_words.union(filtered_embeddding_words))
        filtered_words.sort()
        logger.info(f'Words remaining {len(filtered_words)}')
        filtered_tokenizer = Vocab()
        for word in filtered_words:
            filtered_tokenizer.add_token(word)

        self.training_data.src_tokenizer = filtered_tokenizer
        self.validation_data.src_tokenizer = filtered_tokenizer
        self.testing_data.src_tokenizer = filtered_tokenizer
        self.src_tokenizer = filtered_tokenizer


    def load_word_vec(self):
        if self.embedding_path is None:
            return None
        dataset_vocab = self.src_tokenizer.stoi
        c=0
        print('EMBEDDING PATH!!',self.embedding_path)
        with open(self.embedding_path,'r') as vec_f:
            n,vec_dim = map(int,vec_f.readline().rstrip().split())
            word_vectors = np.random.normal(0, 0.1,(len(dataset_vocab),vec_dim))
            word_vectors[1] = np.zeros(shape = (vec_dim)) #pad index
            for line in vec_f:
                word = line.strip().split()[0]
                if word in dataset_vocab:
                    vec = list(map(float,line.strip().split()[1:]))
                    assert dataset_vocab[word]!=1
                    word_vectors[dataset_vocab[word]] = vec
                    c+=1
        logger.info(f'Loaded {c}/{len(dataset_vocab)} glove vectors')
        return word_vectors


    def collate_fn(self,batch):
        phrase_batch = []
        phrase_len_batch = []
        cmd_batch = []
        cmd_len_batch = []

        max_cmd_len = 0
        for batch_item  in batch:
            (phrases, phrase_len), (cmd,cmd_len) = batch_item
            phrase_batch.append(phrases)
            phrase_len_batch.append(phrase_len)
            cmd_batch.append(cmd)
            cmd_len_batch.append(cmd_len)
            max_cmd_len = max(max_cmd_len,cmd_len)
        
        #PAD ast traversal
        trg_pad_idx = self.trg_tokenizer.pad
        for i,item in enumerate(cmd_batch):
            item_len = cmd_len_batch[i]
            pad_tensor = torch.LongTensor([trg_pad_idx]).repeat(max_cmd_len - item_len)
            cmd_batch[i] = torch.cat([item,pad_tensor])
        
        #PAD phrase and phrase_len
        pad_all_(phrase_len_batch, padding_element=0) #0 is padding element for lengths
        pad_all_(phrase_batch,padding_element=self.src_tokenizer.pad)
        phrase_len_batch = torch.LongTensor(phrase_len_batch).to(self.device) #batch size X max phrase count
        phrase_batch = torch.LongTensor(phrase_batch).to(self.device) #batch size X max phrase count X max phrase length

        cmd_batch = torch.stack(cmd_batch).unsqueeze(-1).to(self.device) #batch x maxlen x 1
        cmd_len_batch = torch.stack(cmd_len_batch).squeeze(-1).to(self.device) #batch
        return (phrase_batch, phrase_len_batch), (cmd_batch,cmd_len_batch)


    def train_dataloader(self):
        if self.multigpu:
            if self.sampler == 'tokens':
                dyn_sampler = DynamicSampler(self.training_data,self.batch_size)
                return DataLoader(self.training_data, collate_fn= self.collate_fn, batch_sampler = dyn_sampler)
            elif self.sampler ==  'sentence':
                static_sampler = StaticSampler(self.training_data,self.batch_size,shuffle=False)
                return DataLoader(self.training_data, collate_fn= self.collate_fn, batch_sampler = static_sampler)
        else:
            return DataLoader(self.training_data, collate_fn= self.collate_fn, batch_size=1)

    def val_dataloader(self):
        if self.multigpu:
            if self.sampler == 'tokens':
                dyn_sampler = DynamicSampler(self.validation_data,self.batch_size)
                return DataLoader(self.validation_data, collate_fn= self.collate_fn, batch_sampler = dyn_sampler)
            elif self.sampler ==  'sentence':
                static_sampler = StaticSampler(self.validation_data,self.batch_size,shuffle=False)
                return DataLoader(self.validation_data, collate_fn= self.collate_fn, batch_sampler = static_sampler)
        else:
            return DataLoader(self.validation_data, collate_fn= self.collate_fn, batch_size=1)

    def test_dataloader(self):
        if self.multigpu:
            if self.sampler == 'tokens':
                dyn_sampler = DynamicSampler(self.testing_data,self.batch_size)
                return DataLoader(self.testing_data, collate_fn= self.collate_fn, batch_sampler = dyn_sampler)
            elif self.sampler ==  'sentence':
                static_sampler = StaticSampler(self.testing_data,self.batch_size,shuffle=False)
                return DataLoader(self.testing_data, collate_fn= self.collate_fn, batch_sampler = static_sampler)
        else:
            return DataLoader(self.testing_data, collate_fn= self.collate_fn, batch_size=1)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--batch',type=int,default=499)
        parser.add_argument('--embedding_path',default = 'Data/WordVectors/MittenVectors/mitten.2000')
        parser.add_argument('--data_base_address',default='Data/parallel_data')
        parser.add_argument('--run_base_address',default = 'run/')
        parser.add_argument('--split',default = '0')
        parser.add_argument('--documentation_path',default = 'Data/documentation/utility_descriptions.csv')
        parser.add_argument('--extra_data_path',nargs='*',default = [])
        parser.add_argument('--process_data', type = str2bool, default = False)
        parser.add_argument('--command_normalised',type = str2bool, default = True)
        parser.add_argument('--sampler',type=str,default='tokens')
        parser.add_argument('--subtree_cut_height',type=int,default=4)
        return parser
