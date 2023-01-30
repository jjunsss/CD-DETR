from xmlrpc.client import Boolean
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from torch.utils.data import DataLoader, ConcatDataset
import datasets.samplers as samplers
import torch
import numpy as np
from typing import Tuple, Collection, Dict, List
import bisect
import random
import matplotlib.pyplot as plt

def Incre_Dataset(Task_Num, args, Incre_Classes):    
    current_classes = Incre_Classes[Task_Num]
    print(f"current_classes : {current_classes}")
    
    if args.Task == 1:
        dataset_train = build_dataset(image_set='train', args=args, class_ids=None) #* Task ID에 해당하는 Class들만 Dataset을 통해서 불러옴
    else: 
        dataset_train = build_dataset(image_set='train', args=args, class_ids=current_classes)
    dataset_val = build_dataset(image_set='val', args=args)
    
    if args.distributed:
        if args.cache_mode:
            sampler_train = samplers.NodeDistributedSampler(dataset_train)
            sampler_val = samplers.NodeDistributedSampler(dataset_val, shuffle=False)
        else:
            sampler_train = samplers.DistributedSampler(dataset_train)
            sampler_val = samplers.DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    print(f"dataset config :{dataset_train}")
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
        Divided_Classes.append([22, 23, 24, 25, 26, 27, 29, 30, 31, 33,34,36, 37, 38, 39, 40,42,43,44,
          45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 57, 58, 59]) #VE
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
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, re_dict):
        self.re_dict = re_dict
        self.keys = list(re_dict.keys())
        values = [v for v in re_dict.values() if v]
        values = np.concatenate(values)
        self.data = np.array(values, dtype = object)        
            
    def __len__(self):
        return len(self.data)
    
    def __repr__(self):
        print(f"trained keys : {self.keys}")    

    def __getitem__(self, idx):
        new_samples, new_targets = self.data[idx]
        
        return new_samples, new_targets, new_samples, new_targets

class BatchMosaicAug(torch.utils.data.Dataset):
    def __init__(self, datasets, CurrentClasses):
        self.Datasets = datasets
        self.labels = [np.zeros((0, 5), dtype=np.float32)] * len(self.Datasets) #[0, xc, yc, w, h]
        self.img_size = 1024 # 만들어질 이미지 크기를 말하는 듯함.
        self.current_classes = CurrentClasses
        im_w = 1024
        im_h = 1024
        
        #!TODO only No Transform image (In Batch Class Augmentation) 
        for index in range(len(self.Datasets)): #TO make Batch Augmentation
            samples, targets, _, _ = self.Datasets[index]
            boxes = targets['boxes'] #xywh
            classes = targets["labels"]
            
            boxes[:, 2] = boxes[:, 0] + boxes[:, 2] #to change x1, y1, x2, y2
            boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
            boxesyolo = []
            for box, cls in zip(boxes, classes):
                x1, y1, x2, y2 = box
                xc, yc, w, h = 0.5*x1/im_w+0.5*x2/im_w, 0.5*y1/im_h+0.5*y2/im_h, abs(x2/im_w-x1/im_w), abs(y2/im_h-y1/im_h)
                boxesyolo.append([cls, xc, yc, w, h])
            self.labels[index] = np.array(boxesyolo)
            
    def load_image(self, index): #torch sample shape : {1, 3, 1066, 800}
        # loads 1 image from dataset, returns img, original hw, resized hw
        img = self.Datasets[index][0].squeeze()
        origin_shape = self.Datasets[index][1]["orig_size"]
        
        w0, h0 = origin_shape[0].item(), origin_shape[1].item()  # orig hw
        return img, (h0, w0), img.shape[:0:-1]  # img, hw_original, hw_resized(height, Width)

    def load_mosaic(index):
        # loads images in a mosaic
        labels4 = []
        s = self.img_size #1024, im_w, im_h : 1024
        xc, yc = [int(random.uniform(s * 0.5, s * 1.5)) for _ in range(2)]  # mosaic center x, y (1024의 0.5 ~ 1.5 사이에 center 생성)
        Mosaic_index_list = []
        while True:
            Mosaic_index = random.randint(0, len(self.labels) - 1)
            Mosaic_index_list.append(Mosaic_index)
            Current_mosaic_index = []
            diff_mosaic_index = []
            if Mosaic_index in set(Mosaic_index_list): #No duplicate
                continue
            
            if self.labels[Mosaic_index][0][0] in self.current_classes:
                if len(Current_mosaic_index) < 3:
                    Current_mosaic_index.append(Mosaic_index)
            else:
                if len(diff_mosaic_index) < 3:
                    diff_mosaic_index.append(Mosaic_index)
        
            if len(Current_mosaic_index) == 3 and len(diff_mosaic_index) == 3:
                break
        
        #*Original Mosaic Training   
        #indices = [index] + [random.randint(0, len(self.labels) - 1) for _ in range(3)]  #여기서 다양한 Index를 추출해야하는데 같은 Task 1개 / 다른 Task 1개 추출해야함(다른은 Rehearsal에서 가져오기)
            
        for i, index in enumerate(indices):
            # Load image
            img, _, (h, w) = load_image(index) #! cv2.imread 통해서 불러옴. 나는 coco 사용하기에 변경해야 함.

            # place img in img4(특정 center point 잡아서 할당)
            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax] / img4가 비어있는 List 
            padw = x1a - x1b
            padh = y1a - y1b

            # Labels
            x = self.labels[index] #! (0, xc, yc, w, h)
            labels = x.copy()
            if x.size > 0:  # Normalized xywh to pixel xyxy format
                labels[:, 1] = w * (x[:, 1] - x[:, 3] / 2) + padw
                labels[:, 2] = h * (x[:, 2] - x[:, 4] / 2) + padh
                labels[:, 3] = w * (x[:, 1] + x[:, 3] / 2) + padw
                labels[:, 4] = h * (x[:, 2] + x[:, 4] / 2) + padh
            labels4.append(labels) #! New labels for trainig

        # Concat/clip labels
        if len(labels4):
            labels4 = np.concatenate(labels4, 0)
            np.clip(labels4[:, 1:] - s / 2, 0, s, out=labels4[:, 1:])  # use with center crop -> my strategy for caculating
            #np.clip(labels4[:, 1:], 0, 2 * s, out=labels4[:, 1:])  # use with random_affine

        # Augment
        img4 = img4[s // 2: int(s * 1.5), s // 2:int(s * 1.5)]  # center crop (WARNING, requires box pruning)
        # img4, labels4 = random_affine(img4, labels4,
        #                               degrees=1.98 * 2,
        #                               translate=0.05 * 2,
        #                               scale=0.05 * 2,
        #                               shear=0.641 * 2,
        #                               border=-s // 2)  # border to remove

        return img4, labels4

#For Rehearsal
def CombineDataset(args, RehearsalData, CurrentDataset, Worker, Batch_size):
    OldDataset = CustomDataset(RehearsalData) #oldDatset[idx]:
    class_ids = CurrentDataset.class_ids
    CombinedDataset = ConcatDataset([OldDataset, CurrentDataset]) #Old : previous, Current : Now
    BatchMosaicAug(CombinedDataset, CurrentDataset)
    
    print(f"current Dataset length : {len(CurrentDataset)} -> Rehearsal + Current length : {len(CombinedDataset)}")
    #CombinedDataset = OldDataset
    if args.distributed:
        if args.cache_mode:
            sampler_train = samplers.NodeDistributedSampler(CombinedDataset)
        else:
            sampler_train = samplers.DistributedSampler(CombinedDataset)
    else:
        sampler_train = torch.utils.data.RandomSampler(CombinedDataset)
        
    batch_sampler_train = torch.utils.data.BatchSampler(
    sampler_train, Batch_size, drop_last=True)
    CombinedLoader = DataLoader(CombinedDataset, batch_sampler=batch_sampler_train,
                        collate_fn=utils.collate_fn, num_workers=Worker,
                        pin_memory=True)
    
    
    return CombinedLoader