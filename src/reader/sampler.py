
import torch
import numpy as np
from main_utils import get_logger
logger = get_logger()


class DynamicSampler(torch.utils.data.distributed.DistributedSampler):
    # handles dynamic batching wrt tokens
    def __init__(self,dataset,batch_size,shuffle=False):
        super(DynamicSampler,self).__init__(dataset,shuffle=shuffle)
        logger.info(f'SAMPLER INFO::DYNAMIC SAMPLER::PARENT::number of replicas={self.num_replicas}, rank={self.rank},\
                    total size={self.total_size}, number of samples={self.num_samples}')
        self.batch_size = batch_size
        if shuffle:
            self.sorted_indices = np.arange(dataset.instance_len.shape[0])
            np.random.shuffle(self.sorted_indices)
        else:
            self.sorted_indices = np.argsort(dataset.instance_len)
        
        self.setup_batches_()
        logger.info(f'SAMPLER INFO::DYNAMIC SAMPLER::CHILD::number of replicas={self.num_replicas}, rank={self.rank},\
                    total size={self.total_size}, number of samples={self.num_samples}')
    
    def setup_batches_(self):
        self.batched_data = []
        current_len = 0
        current_batch = []
        for i,idx in enumerate(self.sorted_indices):
            current_len += self.dataset.instance_len[idx]
            current_batch.append(idx)
            if i==len(self.sorted_indices)-1  or current_len + self.dataset.instance_len[self.sorted_indices[i+1]] > self.batch_size:
                self.batched_data.append(current_batch)
                current_batch = []
                current_len = 0
        
        # duplicate batches if not divisible by num of replicas, else deadlocks
        rem = len(self.batched_data) % self.num_replicas
        if rem != 0:
            rem = self.num_replicas - rem
        self.batched_data += self.batched_data[-rem:]
        assert len(self.batched_data)%self.num_replicas == 0

        self.total_size = len(self.batched_data) # total samples
        self.num_samples =  self.total_size // self.num_replicas #samples per gpu

    def __iter__(self):
        # subsample
        current_gpu_batches = []
        i=self.rank
        total_size = len(self.batched_data)
        while i<total_size:
            current_gpu_batches.append(self.batched_data[i])
            i+=self.num_replicas
        assert self.num_samples == len(current_gpu_batches)
        assert self.total_size == len(self.batched_data)
        return iter(current_gpu_batches)

class StaticSampler(torch.utils.data.distributed.DistributedSampler):
    # handles static batching wrt paired datapoints
    def __init__(self,dataset,batch_size,shuffle=False):
        super(StaticSampler,self).__init__(dataset,shuffle=shuffle)
        logger.info(f'SAMPLER INFO::STATIC SAMPLER::PARENT::number of replicas={self.num_replicas}, rank={self.rank},\
                    total size={self.total_size}, number of samples={self.num_samples}')
        self.batch_size = batch_size
        if shuffle:
            self.sorted_indices = np.arange(dataset.instance_len.shape[0])
            np.random.shuffle(self.sorted_indices)
        else:
            self.sorted_indices = np.argsort(dataset.instance_len)
        
        self.setup_batches_()
        logger.info(f'SAMPLER INFO::STATIC SAMPLER::CHILD::number of replicas={self.num_replicas}, rank={self.rank},\
                    total size={self.total_size}, number of samples={self.num_samples}')
    
    def setup_batches_(self):
        self.batched_data = []
        current_len = 0
        current_batch = []
        for i,idx in enumerate(self.sorted_indices):
            current_len += self.dataset.instance_len[idx]
            current_batch.append(idx)
            if current_len >= self.batch_size or i==len(self.sorted_indices)-1:
                self.batched_data.append(current_batch)
                current_len = 0
        
        # duplicate batches if not divisible by num of replicas, else deadlocks
        rem = len(self.batched_data) % self.num_replicas
        if rem != 0:
            rem = self.num_replicas - rem
        self.batched_data += self.batched_data[-rem:]
        assert len(self.batched_data)%self.num_replicas == 0

        self.total_size = len(self.batched_data) # total samples
        self.num_samples =  self.total_size // self.num_replicas #samples per gpu

    def __iter__(self):
        # subsample
        current_gpu_batches = []
        i=self.rank
        total_size = len(self.batched_data)
        while i<total_size:
            current_gpu_batches.append(self.batched_data[i])
            i+=self.num_replicas
        assert self.num_samples == len(current_gpu_batches)
        assert self.total_size == len(self.batched_data)
        return iter(current_gpu_batches)