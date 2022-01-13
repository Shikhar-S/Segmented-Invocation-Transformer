# DataReader
from nltk import tree
from nltk.util import print_string
import torch
from torch.utils.data import Dataset
from reader.vocab import Vocab
import numpy as np
import nltk
import pickle
from main_utils import get_logger, timeit

from bashlint import data_tools, nast
from collections import deque

logger = get_logger()

class DataReader(Dataset):
    def __init__(self,\
                paths,\
                subtree_cut_height = 4,\
                src_tokenizer=None,\
                trg_tokenizer = None,\
                extra_data_paths = [],\
                n_examples = -1):
        super(DataReader,self).__init__()
        self.src_path,self.trg_path,self.tree_path = paths
        self.max_trg_length = -1
        self.max_phrases_count = -1
        self.max_phrases_len = -1
        self.subtree_cut_height = subtree_cut_height

        self.n_examples = n_examples

        if src_tokenizer is None:
            self.src_tokenizer = Vocab()
        else:
            self.src_tokenizer = src_tokenizer

        if trg_tokenizer is None:
            self.trg_tokenizer = Vocab()
        else:
            self.trg_tokenizer = trg_tokenizer

        self.read_raw_data(extra_data_paths)


    def replace_args(self,node):
        if node.is_argument():
            node.value = node.arg_type
        for i in range(len(node.children)):
            node.children[i] = self.replace_args(node.children[i])
        return node


    def get_util_flags(self,root):
        queue = deque()
        queue.append(root)
        util_flag = {}
        while len(queue)>0:
            top = queue.popleft()
            if top.is_utility():
                if top.value not in util_flag:
                    util_flag[top.value] = top.get_flags()
                else:
                    util_flag[top.value].extend(top.get_flags())
            for child in top.children:
                queue.append(child)
        return util_flag


    def get_type_value(self,root):
        queue = deque()
        queue.append(root)
        kind_value = {}
        while len(queue)>0:
            top = queue.popleft()
            if top.kind not in kind_value:
                kind_value[top.kind] = set([top.value])
            else:
                kind_value[top.kind].add(top.value)
            for child in top.children:
                queue.append(child)
        return kind_value


    def read_raw_data(self,extra_data_paths):
        '''
        Reads raw data from all files
        '''
        self.tree_list = []

        self.y = []
        self.y_len = []

        #Read nlc2cmd data
        self.read_raw_data_(self.src_path,self.trg_path,self.tree_path,n_examples = self.n_examples)
        #Read extra data
        for src_path,trg_path,tree_path in extra_data_paths:
            self.read_raw_data_(src_path,trg_path,tree_path)

        #Sanity checks
        assert len(self.y) == len(self.y_len)
        for trg_sentence in self.y:
            assert len(trg_sentence) > 0

        #Post processing read data
        self.y_len = np.array(self.y_len)
        self.instance_len = self.y_len

        #Add tokens to vocab
        self.trg_tokenizer.build_dic(self.y)


        self.phrases_list, self.phrases_len_list = [], []
        for tree in self.tree_list:
            self.build_leaf_paths(tree)
            phrases,phrases_len = self.getTreePhrases(tree,self.subtree_cut_height)
            self.phrases_list.append(phrases)
            self.phrases_len_list.append(phrases_len)
            self.max_phrases_count = max(self.max_phrases_count,len(phrases))
            self.max_phrases_len = max(self.max_phrases_len, max(phrases_len))

        #add tree tokens to vocab
        for sentence in self.phrases_list:
            for phrase in sentence:
                for token in phrase:
                    self.src_tokenizer.add_token(token)

        assert len(self.y) == len(self.tree_list)
        #log num tokens read and max length
        logger.info(f"Trg Vocab Size {self.trg_tokenizer.vocab_len}")
        logger.info(f"Src Vocab Size {self.src_tokenizer.vocab_len}")
        logger.info(f"Max number of phrases {self.max_phrases_count},\
                    Largest phrase length {self.max_phrases_len}")

    def read_raw_data_(self,src_path,trg_path,tree_path,n_examples = -1):
        '''
        Reads raw data from file
        '''
        logger.info(f'Reading from {src_path} and {trg_path}')
        #Read target data
        n_read = 0
        with open(trg_path,'r') as f:
            for line in f:
                n_read+=1
                root = self.replace_args(data_tools.bash_parser(line.strip()))
                normalised_command_tokens = ['[SOS]'] +  data_tools.ast2command(root).strip().split() + ['[EOS]']
                self.y.append(normalised_command_tokens)
                self.y_len.append(len(normalised_command_tokens))
                self.max_trg_length = max(self.max_trg_length,self.y_len[-1])
                if n_examples != -1 and n_read == n_examples:
                    break

        #Read source trees
        tree_list = []
        with open(tree_path,'rb') as f:
            tree_list = pickle.load(f)
            if n_examples!=-1:
                tree_list = tree_list[:n_examples]
        assert n_examples==-1 or len(tree_list) == n_examples
        self.tree_list += tree_list

    def getTreePhrases(self,node,cut_height = 4, method = 'dfs'):
        '''
        Get phrases from tree
        '''
        phrases,phrases_len = [],[]
        Q = deque()
        Q.append(node)
        while len(Q)>0:
            if method=='dfs':
                top = Q.pop()
            elif method=='bfs':
                top = Q.popleft()
            if top.height <= cut_height:
                phrase = [leaf_node.value for leaf_node in top.leaf_list]
                phrases.append(phrase)
                phrases_len.append(len(phrase))
            else:
                if method=='dfs':
                    iteration_list = reversed(top.children)
                elif method=='bfs':
                    iteration_list = top.children
                for child in iteration_list:
                    Q.append(child)
        return phrases,phrases_len

    def __len__(self):
        return len(self.y_len)

    def build_leaf_paths(self,current_node):
        '''
        Stores all leaves for each internal node of tree
        '''
        #store all leaves in subtree rooted at current_node in order
        if current_node.height==0: #is leaf
            current_node.leaf_list.append(current_node)

        for child in current_node.children:
            self.build_leaf_paths(child)
            current_node.leaf_list += child.leaf_list

    def get_phrases_tokenized(self,vocab, phrase_list):
        '''
        Returns a list of phrases tokenized by vocab
        '''
        phrase_id = []
        for phrase in phrase_list:
            phrase_id.append([vocab.get_id(token) for token in phrase])
        return phrase_id

    def print_tree(self,node,tabs = 0):
        '''
        Prints the tree in a readable format
        '''
        print('- '*tabs,node.value, node.height)
        for child in node.children:
            self.print_tree(child,tabs+1)

    def __getitem__(self, idx):
        '''
        Returns a tuple of (src_tokens, trg_tokens, src_len, trg_len, tree, phrases, phrases_len)
        '''
        cmd = self.y[idx]
        cmd_len = min(self.max_trg_length,self.y_len[idx])

        y_tokens = []
        for word in cmd:
            token_id = self.trg_tokenizer.get_id(word.strip())
            y_tokens.append(token_id)

        y_tokens = y_tokens[:self.max_trg_length]
        cmd_tensor = torch.LongTensor(y_tokens)
        cmd_len = torch.LongTensor([cmd_len])

        phrases_tokenized=self.get_phrases_tokenized(self.src_tokenizer, self.phrases_list[idx])
        phrase_lengths = self.phrases_len_list[idx]

        return (phrases_tokenized, phrase_lengths), (cmd_tensor, cmd_len)