# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from codes in torch.utils.data.distributed
# ------------------------------------------------------------------------

import os
import math
import torch
import torch.distributed as dist
from torch.utils.data.sampler import Sampler


class DistributedSampler(Sampler): # FOR DDP Training, we control total Dataset
    """Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, dataset, num_replicas=None, rank=None, local_rank=None, local_size=None, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size() # 4
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas)) # 140000/ 4 = 35000 개씩
        self.total_size = self.num_samples * self.num_replicas # 하나의 gpu에 올라갈 이미지 개수 = 35000 / gpu 개수 = 4 
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch
            g = torch.Generator() #난수 생성 Generator
            g.manual_seed(self.epoch) #epoch마다 새로운 난수가 생성되어짐
            indices = torch.randperm(len(self.dataset), generator=g).tolist() #batch를 섞는 행위
        else:
            indices = torch.arange(len(self.dataset)).tolist()  # 140000

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample 데이터 14만개를 나눠서 살펴봄(각 gpu 마다 35000장의 이미지를 살펴보는 것)
        
        offset = self.num_samples * self.rank
        indices = indices[offset : offset + self.num_samples]
        #TODO : iter 자체를 비율을 맞춰서 할당하면 ..? 각 에폭마다 어차피 달라질 것이고, Random을 설정해두면 더 좋을 듯?
        #TODO : 여기서 분할된 인덱스 번호를 사용해서..?
        assert len(indices) == self.num_samples


        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class NodeDistributedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, dataset, num_replicas=None, rank=None, local_rank=None, local_size=None, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
            
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
            
        if local_rank is None:
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
            
        if local_size is None:
            local_size = int(os.environ.get('LOCAL_SIZE', 1))
            print(f"local_size: {local_size}")
            
        self.dataset = dataset
        self.shuffle = shuffle
        self.num_replicas = num_replicas
        self.num_parts = 4 # local_size
        self.rank = rank
        self.local_rank = local_rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

        self.total_size_parts = self.num_samples * self.num_replicas // self.num_parts

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()
        indices = [i for i in indices if i % self.num_parts == self.local_rank]

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size_parts - len(indices))]
        assert len(indices) == self.total_size_parts

        # subsample
        print(f'*******************3: {self.num_replicas} {self.num_parts}')
        indices = indices[self.rank // self.num_parts:self.total_size_parts:self.num_replicas // self.num_parts]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
