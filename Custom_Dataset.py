from xmlrpc.client import Boolean
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from torch.utils.data import DataLoader, ConcatDataset
import datasets.samplers as samplers
import torch
import numpy as np
from termcolor import colored


def Incre_Dataset(Task_Num, args, Incre_Classes, extra_dataset = False):    
    current_classes = Incre_Classes[Task_Num]
    print(f"current_classes : {current_classes}")
    all_classes = sum(Incre_Classes[:Task_Num+1], []) # ALL : old task clsses + new task clsses(after training, soon to be changed)\
          
    if not extra_dataset and not args.eval:
        # For real model traning
        dataset_train = build_dataset(image_set='train', args=args, class_ids=current_classes)
            
    elif extra_dataset and not args.eval:
        # For generating buffer with whole dataset
        # previous classes are used to generate buffer of all classe before New task dataset
        print(colored(f"extra dataset config..", "blue", "on_yellow"))
        print(colored(f"collecte class categories in extre epoch: {all_classes}", "blue", "on_yellow"))
        # necessary for calling current task dataset, buffer is already collected from previous process
        dataset_train = build_dataset(image_set='extra', args=args, class_ids=current_classes) 
    
    if args.eval :
        tgt = current_classes if args.LG else all_classes
        print(colored(f"evaluation check : {tgt}", "blue", "on_yellow"))
        dataset_val = build_dataset(image_set='val', args=args, class_ids=tgt)            
        
    if args.distributed:
        if args.cache_mode:
            if not args.eval:
                sampler_train = samplers.NodeDistributedSampler(dataset_train)
            else:
                sampler_val = samplers.NodeDistributedSampler(dataset_val, shuffle=False)
        else:
            if not args.eval:
                sampler_train = samplers.DistributedSampler(dataset_train)
            else:
                sampler_val = samplers.DistributedSampler(dataset_val, shuffle=True)
    else:
        if not args.eval:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        
    if not args.eval:
        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, args.batch_size, drop_last=True)
        data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                    collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                    pin_memory=True, prefetch_factor=args.prefetch)
    else:
        data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                     drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                     pin_memory=True, prefetch_factor=args.prefetch)
        return dataset_val, data_loader_val, sampler_val, all_classes
    
    if extra_dataset :
        current_classes = all_classes
        
    return dataset_train, data_loader_train, sampler_train, current_classes

def make_class(test_file):
    #####################################
    ########## !! Edit here !! ##########
    #####################################
    class_dict = {
        'file_name': ['did', 'pz', 've'],
        'class_idx': [
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21], # DID
            [28, 32, 35, 41, 56], # photozone
            [24, 29, 30, 39, 40, 42] # 야채칸 중 일부(mAP 높은 일부)
        ]
    }
    #####################################
    
    # case_1) file name에 VE가 포함되어 있지 않은 경우
    if test_file.lower() in ['2021', 'multisingle', '10test']:
        test_file = 've' + test_file
    # case_2) 혼합 데이터셋
    if '+' in test_file:
        task_list = test_file.split('+')
        tmp = []
        for task in task_list:
            idx = [name in task.lower() for name in class_dict['file_name']].index(True)
            tmp.append(class_dict['class_idx'][idx])
        res = sum(tmp, [])
        return res  # early return
    
    idx = [name in test_file.lower() for name in class_dict['file_name']].index(True)
    return class_dict['class_idx'][idx]


def DivideTask_for_incre(args, Task_Counts: int, Total_Classes: int, DivisionOfNames: Boolean, eval_config=False, test_file_list=None):
    '''
        DivisionofNames == True인 경우 Task_Counts는 필요 없어짐 Domain을 기준으로 class task가 자동 분할
        False라면 Task_Counts, Total_Classes를 사용해서 적절하게 분할
        #Task : 테스크의 수
        #Total Class : 총 클래스의 수
        #DivisionOfNames : Domain을 사용해서 분할
    '''
    if DivisionOfNames is True:
        Divided_Classes = []
        if test_file_list is not None:
            for test_file in test_file_list:
                Divided_Classes.append(
                    make_class(test_file)
                )
        else:
            if args.LG:
                Divided_Classes.append([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,]) # DID + PZ
                Divided_Classes.append([28, 32, 35, 41, 56]) # PZ 
                # Divided_Classes.append([24, 29, 30, 39, 40, 42]) # custom VE
            else:                
                Divided_Classes.append([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45]) # 45
                Divided_Classes.append([46, 47, 48, 49, 50, 51, 52, 53, 54, 55]) # real classes 10
                Divided_Classes.append([56, 57, 58, 59, 60, 61, 62, 63, 64, 65]) # real classes 10
                Divided_Classes.append([66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]) # real classes 10
                Divided_Classes.append([80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90]) # real classes 10
                
            if args.divide_ratio == '4040':
                T1 = Divided_Classes[0]
                T2 = [item for sublist in Divided_Classes[1:] for item in sublist]
                Divided_Classes_detail = [T1, T2]
                print(colored(f"Divided_Classes :{Divided_Classes_detail}", "blue", "on_yellow"))
                
            if args.divide_ratio == '402020':
                T1 = Divided_Classes[0]
                T2 = [item for sublist in Divided_Classes[1:3] for item in sublist]
                T3 = [item for sublist in Divided_Classes[3:] for item in sublist]
                Divided_Classes_detail = [T1, T2, T3]
                print(colored(f"Divided_Classes :{Divided_Classes_detail}", "blue", "on_yellow"))
                
            if args.divide_ratio == '4010101010':
                T1 = Divided_Classes[0]
                T2 = [item for sublist in Divided_Classes[1] for item in sublist]
                T3 = [item for sublist in Divided_Classes[2] for item in sublist]
                T4 = [item for sublist in Divided_Classes[3] for item in sublist]
                T5 = [item for sublist in Divided_Classes[4] for item in sublist]
                Divided_Classes_detail = [T1, T2, T3, T4, T5]
                print(colored(f"Divided_Classes :{Divided_Classes_detail}", "blue", "on_yellow"))

            elif args.divide_ratio == "7010":
                T1 = [item for sublist in Divided_Classes[:-1] for item in sublist]
                T2 = Divided_Classes[-1]
                Divided_Classes_detail = [T1, T2]
                print(colored(f"Divided_Classes :{Divided_Classes_detail}", "blue", "on_yellow"))
            
            elif args.divide_ratio == "8000":
                T1 = [item for sublist in Divided_Classes[:] for item in sublist]
                T2 = []
                Divided_Classes_detail = [T1, T2]
                print(colored(f"Divided_Classes :{Divided_Classes_detail}", "blue", "on_yellow"))
                
                
        msg = f"{'='*35} Entire Divided Classes {'='*35}\n{Divided_Classes}"
        print(colored(msg, 'red'))
        return Divided_Classes_detail

    # For auto division dataset(T2 training) (40-40 and), (70-10 or 10-70) to be used better performance setting
    # classes = [idx+1 for idx in range(Total_Classes)] # for COCO
    classes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, \
                56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90] # for absence coco index
    Total_Classes = len(classes)
    print(colored(f"total classes :{Total_Classes}", "blue", "on_yellow"))
    
    if args.divide_ratio != '40':
        t1_class_num = int(args.divide_ratio[:2])
        Rest_Classes_num = 0
        Divided_Classes = [classes[:t1_class_num], classes[t1_class_num:]]
        print(colored(f"Divided_Classes :{Divided_Classes}", "blue", "on_yellow"))
    else:
        # equal distribution
        #TODO: Plz have to update
        Task = int(Total_Classes / Task_Counts)
        Rest_Classes_num = Total_Classes % Task_Counts
        
        start = 0
        end = Task
        Divided_Classes = []
        
        for _ in range(Task_Counts):
            Divided_Classes.append(classes[start:end])
            start += Task
            end += Task
        
    # if remaindar exists.
    if Rest_Classes_num != 0:
        Rest_Classes = classes[-Rest_Classes_num:]
        Divided_Classes[-1].extend(Rest_Classes)
    
    if eval_config :
        classes = [idx+1 for idx in range(args.Test_Classes)]
        Divided_Classes = [classes]
        
    return Divided_Classes

#현재 (Samples, Targets)의 정보를 가진 형태로 데이터가 구성되어 있음(딕셔너리로 각각의 Class 정보를 가진 채로 구성됨)
#참고로 Samples -> NestedTensor, Target -> List 형태로 구성되어 있음 다만 1개의 

from collections import defaultdict
import numpy as np

def weight_dataset(args, re_dict):

    if  args.Sampling_strategy != 'icarl':
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

    else:
        keys = []
        weights = []
        for cls, val in re_dict.items():
            img_ids = np.array(val[1])
            keys.extend(list(img_ids[:, 0].astype(int)))
            weights.extend(list(img_ids[:,1]))

        indices = np.argsort(weights)
        keys = [keys[i] for i in indices]
        weights = [weights[i] for i in indices]
    
    return keys, weights


def img_id_config_no_circular_training(args, re_dict):
    if args.Sampling_strategy == 'icarl':
        keys = []
        for cls, val in re_dict.items():
            img_ids = np.array(val[1])
            keys.extend(list(img_ids[:, 0].astype(int)))
        return keys
    else:
        return list(re_dict.keys())



import copy
from sklearn.preprocessing import QuantileTransformer
import numpy as np
class CustomDataset(torch.utils.data.Dataset):
    '''
        replay buffer configuration
        1. Weight based Circular Experience Replay (WCER)
        2. Fisher based Circular Experience Replay (FCER)
        3. Fisher based ER
    '''
    def __init__(self, args, re_dict, old_classes, fisher_dict = None):
        self.re_dict = copy.deepcopy(re_dict)
        self.old_classes = old_classes
        if args.CER == "weight" and args.AugReplay:
            self.fisher_softmax_weights = None
            self.keys, self.weights = weight_dataset(args, re_dict)
            self.datasets = build_dataset(image_set='train', args=args, class_ids=self.old_classes, img_ids=self.keys)
        elif args.CER == "fisher" and args.AugReplay:
            self.weights = None
            self.keys = list(self.re_dict.keys())
            self.datasets = build_dataset(image_set='train', args=args, class_ids=self.old_classes, img_ids=self.keys)
            fisher_values = torch.tensor(list(fisher_dict.values()))
            scaled_fisher_values = self.scaling(fisher_values)
            
            # Calculate softmax weights
            self.fisher_softmax_weights = torch.softmax(scaled_fisher_values, dim=0)
        else :
            self.weights = None
            self.fisher_softmax_weights = None
            self.keys = img_id_config_no_circular_training(args, re_dict)
            self.datasets = build_dataset(image_set='train', args=args, class_ids=self.old_classes, img_ids=self.keys)
            

    def __len__(self):
        return len(self.datasets)
    
    def __repr__(self):
        print(f"Data key presented in buffer : {self.old_classes}")    

    def __getitem__(self, idx):
        samples, targets, new_samples, new_targets = self.datasets[idx]

        return samples, targets, new_samples, new_targets
    

    def scaling(self, tensor):
        # Quantile Scaling
        qt = QuantileTransformer()
        
        # Transform tensor to numpy array for scaling
        tensor_np = tensor.cpu().detach().numpy()
        
        # The fit_transform expects 2D data, so we need to add extra dim
        tensor_np = np.expand_dims(tensor_np, axis=1)
        
        # Fit and transform data
        scaled_array = qt.fit_transform(tensor_np)
        
        # Convert back to tensor
        scaled_tensor = torch.from_numpy(scaled_array).float()
        
        # Remove the extra dimension
        scaled_tensor = torch.squeeze(scaled_tensor)

        return scaled_tensor
import copy
class ExtraDataset(torch.utils.data.Dataset):
    '''
        replay buffer configuration
        1. Weight based Circular Experience Replay (WCER)
        2. Fisher based Circular Experience Replay (FCER)
        3. Fisher based ER
    '''
    def __init__(self, args, re_dict, old_classes):
        self.re_dict = copy.deepcopy(re_dict)
        self.old_classes = old_classes
        self.keys = list(self.re_dict.keys())
        self.datasets = build_dataset(image_set='extra', args=args, class_ids=self.old_classes, img_ids=self.keys)
            
    def __len__(self):
        return len(self.datasets)
    
    def __repr__(self):
        print(f"Data key presented in buffer : {self.old_classes}")    

    def __getitem__(self, idx):
        samples, targets, new_samples, new_targets = self.datasets[idx]

        return samples, targets, new_samples, new_targets

import random
import collections

import torch.distributed as dist
class NewDatasetSet(torch.utils.data.Dataset):
    def __init__(self, args, CCB_augmentation, datasets, OldDataset, OldDataset_weights, fisher_weight, AugReplay=False, Mosaic=False):
        self.args = args
        self.Datasets = datasets #now task
        self.Rehearsal_dataset = OldDataset
        self.AugReplay = AugReplay
        self.OldDataset_weights = OldDataset_weights
        self.fisher_weights = fisher_weight
        self.Mosaic = Mosaic #for mosaic augmentation
        self.img_size = (480, 640) #for mosaic augmentation
        if self.AugReplay == True:
            self.old_length = len(self.Rehearsal_dataset) if dist.get_world_size() == 1 else int(len(self.Rehearsal_dataset) // dist.get_world_size()) # 4
        if self.Mosaic == True: 
            self.old_length = len(OldDataset)
            self._CCB = CCB_augmentation(self.img_size, self.args.Continual_Batch_size)

            
    def __len__(self):
        return len(self.Datasets)

    def __getitem__(self, index): 
        img, target, origin_img, origin_target = self.Datasets[index] #No normalize pixel, Normed Targets
        if self.AugReplay == True :
            if self.args.CER == "fisher": # fisher CER
                index = np.random.choice(np.arange(len(self.Rehearsal_dataset)), p=self.fisher_weights.numpy())
                O_img, O_target, _, _ = self.Rehearsal_dataset[index] #No shuffle because weight sorting.
                return img, target, origin_img, origin_target, O_img, O_target
            elif self.args.CER == "weight": # weight CER
                index = np.random.choice(np.arange(len(self.Rehearsal_dataset)), p=self.OldDataset_weights)
                O_img, O_target, _, _ = self.Rehearsal_dataset[index] #No shuffle because weight sorting.
                return img, target, origin_img, origin_target, O_img, O_target
            elif self.args.CER == "original": # original CER
                if index > (len(self.Rehearsal_dataset)-1):
                    index = index % len(self.Rehearsal_dataset)
                    O_img, O_target, _, _ = self.Rehearsal_dataset[index] #No shuffle because weight sorting.
                    return img, target, origin_img, origin_target, O_img, O_target
                O_img, O_target, _, _ = self.Rehearsal_dataset[index]
                return img, target, origin_img, origin_target, O_img, O_target
    
        if self.Mosaic == True :
            Current_mosaic_index = self._Mosaic_index()
            image_list = []
            target_list = []
            for index in Current_mosaic_index:
                _, _ , o_img, otarget = self.Datasets[index] #Numpy image / torch.tensor
                image_list.append(o_img)
                target_list.append(otarget)
            
            if self.args.Continual_Batch_size == 2:
                Cur_img, Cur_lab = self._CCB(image_list, target_list)
                return img, target, origin_img, origin_target, Cur_img, Cur_lab #cur_img, cur_lab = mosaic images, mosaic labels
            
            if self.args.Continual_Batch_size == 3:
                Cur_img, Cur_lab, Dif_img, Dif_lab = self._CCB(image_list, target_list)
                return img, target, origin_img, origin_target, Cur_img, Cur_lab, Dif_img, Dif_lab
            
        return img, target, origin_img, origin_target
    
    def _Mosaic_index(self): #* Done
        '''
            Only Mosaic index printed 
            index : index in dataset (total dataset = old + new )
            #TODO : count class variables need !! 
        '''
        #*Curretn Class augmentation / Other class AUgmentation
        
        Rehearsal_index = random.choices(range(self.old_length), k=2) #TODO : sampling method change.
        current_index = random.choices(range(self.old_length, len(self.Datasets)), k=2) #TODO : sampling method change.
            
        Rehearsal_index.insert(0, current_index[0])
        Rehearsal_index.insert(0, current_index[1])
        return random.sample(Rehearsal_index, len(Rehearsal_index))
    
#For Rehearsal
from Custom_augmentation import CCB
def CombineDataset(args, RehearsalData, CurrentDataset, 
                   Worker, Batch_size, old_classes, fisher_dict=None, MixReplay = None):
    '''
        MixReplay arguments is only used in MixReplay. If It is not args.MixReplay, So
        you can ignore this option.
    '''
    OldDataset = CustomDataset(args, RehearsalData, old_classes, fisher_dict=fisher_dict) #oldDatset[idx]:
    OldDataset_weights = OldDataset.weights
    old_fisher_weight = OldDataset.fisher_softmax_weights
    
    if args.MixReplay and MixReplay == "Original" :
        CombinedDataset = ConcatDataset([OldDataset, CurrentDataset])
        NewTaskdataset = NewDatasetSet(args, CCB, CombinedDataset, OldDataset, OldDataset_weights, old_fisher_weight, AugReplay=False)
         
    elif args.MixReplay and MixReplay == "AugReplay" : 
        CombinedDataset = ConcatDataset([OldDataset, CurrentDataset])
        NewTaskdataset = NewDatasetSet(args, CCB, CombinedDataset, OldDataset, OldDataset_weights, old_fisher_weight, AugReplay=True, Mosaic=False) \
    
        if args.distributed:
            if args.cache_mode:
                sampler_train = samplers.NodeDistributedSampler(NewTaskdataset)
            else:
                sampler_train = samplers.CustomDistributedSampler(NewTaskdataset, OldDataset, old_fisher_weight, shuffle=True)
        else:
            sampler_train = torch.utils.data.RandomSampler(NewTaskdataset)
            
        batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, Batch_size, drop_last=True)
        CombinedLoader = DataLoader(NewTaskdataset, batch_sampler=batch_sampler_train,
                        collate_fn=utils.collate_fn, num_workers=Worker,
                        pin_memory=True, prefetch_factor=args.prefetch) #worker_init_fn=worker_init_fn, persistent_workers=args.AugReplay)
        return NewTaskdataset, CombinedLoader, sampler_train
        
    if args.AugReplay and not args.MixReplay :
        '''
            circular training process
            new_dataset : 4 gpu devide 
            buffer_datset : 4 gpu devide and random sampler processing
        '''
        NewTaskdataset = NewDatasetSet(args, CCB, CurrentDataset, OldDataset, OldDataset_weights, old_fisher_weight, AugReplay=True, Mosaic=False)
    
    elif not args.AugReplay and not args.MixReplay and args.Mosaic :
        # mosaic dataset configuration
        CombinedDataset = ConcatDataset([OldDataset, CurrentDataset])
        NewTaskdataset = NewDatasetSet(args, CCB, CombinedDataset, OldDataset, OldDataset_weights, old_fisher_weight, AugReplay=False, Mosaic=True) \
            
    elif not args.AugReplay and not args.MixReplay and not args.Mosaic:
        CombinedDataset = ConcatDataset([OldDataset, CurrentDataset])
        NewTaskdataset = NewDatasetSet(args, CCB, CombinedDataset, OldDataset, OldDataset_weights, old_fisher_weight, AugReplay=False) 
        
    print(f"current Dataset length : {len(CurrentDataset)}")
    print(f"Total Dataset length : {len(CurrentDataset)} +  old dataset length : {len(OldDataset)}")
    print(colored(f"********** sucess combined Dataset ***********", "blue"))
    
    if args.distributed:
        if args.cache_mode:
            sampler_train = samplers.NodeDistributedSampler(NewTaskdataset)
        else:
            sampler_train = samplers.DistributedSampler(NewTaskdataset, shuffle=True)
    else:
        sampler_train = torch.utils.data.RandomSampler(NewTaskdataset)
        
    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, Batch_size, drop_last=True)
    CombinedLoader = DataLoader(NewTaskdataset, batch_sampler=batch_sampler_train,
                    collate_fn=utils.collate_fn, num_workers=Worker,
                    pin_memory=True, prefetch_factor=args.prefetch) #worker_init_fn=worker_init_fn, persistent_workers=args.AugReplay)

    return NewTaskdataset, CombinedLoader, sampler_train


    
def IcarlDataset(args, single_class:int):
    '''
        For initiating prototype-mean of the feature of corresponding, single class-, dataset composed to single class is needed.
    '''
    dataset = build_dataset(image_set='train', args=args, class_ids=[single_class])
    if len(dataset) == 0:
        return None, None, None
    
    if args.distributed:
        if args.cache_mode:
            sampler = samplers.NodeDistributedSampler(dataset)
        else:
            sampler = samplers.DistributedSampler(dataset)  
    else:
        sampler = torch.utils.data.RandomSampler(dataset)
        
    batch_sampler = torch.utils.data.BatchSampler(sampler, args.batch_size, drop_last=True)
    data_loader = DataLoader(dataset, batch_sampler=batch_sampler,
                                collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                pin_memory=True)
    
    return dataset, data_loader, sampler


def fisher_dataset_loader(args, RehearsalData, old_classes):
    buffer_dataset = ExtraDataset(args, RehearsalData, old_classes)
    
    sampler_train = torch.utils.data.SequentialSampler(buffer_dataset)
        
    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, batch_size=1, drop_last=False)
    
    data_loader = DataLoader(buffer_dataset, batch_sampler=batch_sampler_train,
                                collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                pin_memory=True, prefetch_factor=args.prefetch)
    
    return data_loader