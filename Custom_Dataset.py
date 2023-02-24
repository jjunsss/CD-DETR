from xmlrpc.client import Boolean
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from torch.utils.data import DataLoader, ConcatDataset
import datasets.samplers as samplers
import torch
import numpy as np
import random
import albumentations as A
from util.box_ops import box_cxcywh_to_xyxy_resize, box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from Custom_augmentation import CCB
from torch.utils.data.sampler import SubsetRandomSampler

def Incre_Dataset(Task_Num, args, Incre_Classes):    
    current_classes = Incre_Classes[Task_Num]
    print(f"current_classes : {current_classes}")
    
    if len(Incre_Classes) == 1:
        dataset_train = build_dataset(image_set='train', args=args, class_ids=None) #* Task ID에 해당하는 Class들만 Dataset을 통해서 불러옴
    else: 
        if Task_Num == 0 : #* First Task training
            dataset_train = build_dataset(image_set='train', args=args, class_ids=current_classes)
        else:
            dataset_train = build_dataset(image_set='train', args=args, class_ids=current_classes)
    dataset_val = build_dataset(image_set='val', args=args)
    
    if args.distributed:
        if args.cache_mode:
            sampler_train = samplers.NodeDistributedSampler(dataset_train)
            sampler_val = samplers.NodeDistributedSampler(dataset_val, shuffle=False)
        else:
            sampler_train = samplers.DistributedSampler(dataset_train)
            sampler_val = samplers.DistributedSampler(dataset_val, shuffle=True)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        
    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                pin_memory=True)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                 pin_memory=True)
    
    return dataset_train, data_loader_train, sampler_train, current_classes


def DivideTask_for_incre(Task_Counts: int, Total_Classes: int, DivisionOfNames: Boolean):
    '''
        DivisionofNames == True인 경우 Task_Counts는 필요 없어짐 Domain을 기준으로 class task가 자동 분할
        False라면 Task_Counts, Total_Classes를 사용해서 적절하게 분할
        #Task : 테스크의 수
        #Total Class : 총 클래스의 수
        #DivisionOfNames : Domain을 사용해서 분할
    '''
    if DivisionOfNames is True:
        Divided_Classes = []
        Divided_Classes.append([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]) #DID
        Divided_Classes.append([28, 32, 35, 41, 56]) #photozone
        Divided_Classes.append([22, 23, 24, 25, 26, 27, 29, 30, 31, 33,34,36, 37, 38, 39, 40,42,43,44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 57, 58, 59]) #VE
        return Divided_Classes
    
    classes = [idx+1 for idx in range(Total_Classes)]
    Task = int(Total_Classes / Task_Counts)
    Rest_Classes_num = Total_Classes % Task_Counts
    
    start = 0
    end = Task
    Divided_Classes = []
    for _ in range(Task_Counts):
        Divided_Classes.append(classes[start:end])
        start += Task
        end += Task
    if Rest_Classes_num != 0:
        Rest_Classes = classes[-Rest_Classes_num:]
        Divided_Classes[-1].extend(Rest_Classes)
    
    return Divided_Classes

#현재 (Samples, Targets)의 정보를 가진 형태로 데이터가 구성되어 있음(딕셔너리로 각각의 Class 정보를 가진 채로 구성됨)
#참고로 Samples -> NestedTensor, Target -> List 형태로 구성되어 있음 다만 1개의 

from collections import defaultdict
import numpy as np

def weight_dataset(re_dict):
    index_counts = defaultdict(int)

    for value in re_dict.values():
        for index in value[1]:
            index_counts[index] += 1

    sorted_classes = sorted(index_counts.items(), key=lambda item : item[0], reverse=False)
    temp = np.array(sorted_classes, dtype=np.float32)
    sum_value = np.sum(temp[:, 1])
    temp[:, 1] /= sum_value

    weight_dict = {}
    for key, value in temp:
        weight_dict[int(key)] = value
        
    for key, value in re_dict.items():
        sumvalue = np.sum([weight_dict[class_idx] for class_idx in value[1]])
        temp_dict_value = re_dict[key]
        temp_dict_value.append(sumvalue)
        re_dict[key] = temp_dict_value
        
    re_dict = dict(sorted(re_dict.items(), key=lambda item : item[1][-1], reverse=True))
    keys = list(re_dict.keys())
    value_array = np.array(list(re_dict.values()), dtype=object)
    weights = value_array[:, -1].tolist()
    
    return keys, weights

import copy
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, args, re_dict, old_classes):
        self.re_dict = copy.deepcopy(re_dict)
        self.old_classes = old_classes
        self.keys, self.weights = weight_dataset(re_dict)
        self.datasets = build_dataset(image_set='train', args=args, class_ids=self.old_classes, img_ids=self.keys)
        
    def __len__(self):
        return len(self.datasets)
    
    def __repr__(self):
        print(f"Data key presented in buffer : {self.old_classes}")    

    def __getitem__(self, idx):
        samples, targets, new_samples, new_targets = self.datasets[idx]

        return samples, targets, new_samples, new_targets


    
class BatchMosaicAug(torch.utils.data.Dataset):
    def __init__(self, datasets, CCB_augmentation, old_length, OldDataset_weights, Mosaic=False, Continual_Batch=2):
        self.Datasets = datasets
        self.Rehearsal_dataset = datasets.datasets[0]
        self.Confidence = 0
        self.Mosaic = Mosaic
        self.img_size = (960, 1280) #Height, Width
        self.old_length = old_length
        self.Continual_Batch = Continual_Batch
        self.OldDataset_weights = OldDataset_weights
        if self.Mosaic == True: 
            #self._CCB = CCB_augmentation(self.Datasets,  self.Rehearsal_dataset, self.Current_dataset, self.img_size)
            self._CCB = CCB_augmentation(self.img_size, self.Continual_Batch)
        
    def __len__(self):
            return len(self.Datasets)    
        
    def __getitem__(self, index):
        img, target, origin_img, origin_target = self.Datasets[index] #No normalize pixel, Normed Targets

        if self.Mosaic == True :
            Current_mosaic_index = self._Mosaic_index(index,)
            image_list = []
            target_list = []
            for index in Current_mosaic_index:
                _, _ , o_img, otarget = self.Datasets[index] #Numpy image / torch.tensor
                image_list.append(o_img)
                target_list.append(otarget)
            
            if self.Continual_Batch == 2:
                Cur_img, Cur_lab = self._CCB(image_list, target_list)
                return img, target, origin_img, origin_target, Cur_img, Cur_lab
            
            if self.Continual_Batch == 3:
                Cur_img, Cur_lab, Dif_img, Dif_lab = self._CCB(image_list, target_list)
                return img, target, origin_img, origin_target, Cur_img, Cur_lab, Dif_img, Dif_lab
        else:
            return img, target, origin_img, origin_target
    
    def _Mosaic_index(self, index): #* Done
        '''
            Only Mosaic index printed 
            index : index in dataset (total dataset = old + new )
            #TODO : count class variables need !! 
        '''
        #*Curretn Class augmentation / Other class AUgmentation
        assert self.old_length == len(self.OldDataset_weights)
        
        Rehearsal_index = random.choices(range(self.old_length), weights=self.OldDataset_weights, k=3) #TODO : sampling method change.
            
        #Mosaic_index.insert(0, index)
        Rehearsal_index.insert(0, index)
        print(f"mosaic index : {Rehearsal_index}")
        return random.sample(Rehearsal_index, len(Rehearsal_index))
    
#For Rehearsal
def CombineDataset(args, RehearsalData, CurrentDataset, Worker, Batch_size, old_classes):
    OldDataset = CustomDataset(args, RehearsalData, old_classes) #oldDatset[idx]:
    Old_length = len(OldDataset)
    OldDataset_weights = OldDataset.weights
    CombinedDataset = ConcatDataset([OldDataset, CurrentDataset]) #Old : previous, Current : Now
    MosaicBatchDataset = BatchMosaicAug(CombinedDataset, CCB, Old_length, OldDataset_weights, args.Mosaic, args.Continual_Batch_size) #* if Mosaic == True -> 1 batch(divided three batch/ False -> 3 batch (only original)
    print(f"current Dataset length : {len(CurrentDataset)} -> Rehearsal + Current length : {len(MosaicBatchDataset)}")
    print(f"********** sucess combined Dataset ***********")
    
    print(MosaicBatchDataset[0])
    if args.distributed:
        if args.cache_mode:
            sampler_train = samplers.NodeDistributedSampler(MosaicBatchDataset)
        else:
            sampler_train = samplers.DistributedSampler(MosaicBatchDataset, shuffle=True)
    else:
        sampler_train = torch.utils.data.RandomSampler(MosaicBatchDataset)
    
    def worker_init_fn(worker_id):
        torch.manual_seed(worker_id)
        
    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, Batch_size, drop_last=True)
    CombinedLoader = DataLoader(MosaicBatchDataset, batch_sampler=batch_sampler_train,
                        collate_fn=utils.collate_fn, num_workers=Worker,
                        pin_memory=True, prefetch_factor=4, worker_init_fn=worker_init_fn, persistent_workers=args.Mosaic)
    
    
    return MosaicBatchDataset, CombinedLoader, sampler_train