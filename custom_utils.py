from ast import arg
from pyexpat import model
from sched import scheduler
from xmlrpc.client import Boolean
import torch
import util.misc as utils
import numpy as np
from typing import Tuple, Dict, List, Optional
import os


def decompose(func):
    def wrapper(no_use_count: int, samples: utils.NestedTensor, targets: Dict, 
                origin_samples: utils.NestedTensor, origin_targets: Dict ,used_number: List) \
            -> Tuple[torch.Tensor, list, bool]:
        output = func(no_use_count, samples, targets, origin_samples, origin_targets, used_number) #Batch 3개라고 가정하고 문제풀이하기
        batch_size = output[0]
        no_use = output[1]
        samples = output[2]
        targets = output[3]
        origin_samples = output[4]
        origin_targets = output[5]
        yes_use = output[6]
        
        if batch_size == 2: #! used to divide rehearsal or main Trainer
            if no_use == 2:
                return samples, targets, origin_samples, origin_targets, False
            if no_use == 1:
                new_targets = []
                origin_new_targets = []
                useit0 = yes_use[0]
                ten, mask = samples.decompose()
                origin_ten = origin_samples
                if ten.shape[0] > 1 :
                    ten0 = torch.unsqueeze(ten[useit0], 0)
                    mask0 = torch.unsqueeze(mask[useit0], 0)
                    samples = utils.NestedTensor(ten0, mask0)
                    new_targets.append(targets[useit0])
                    targets = [{k: v for k, v in t.items()} for t in new_targets]
                    
                origin_ten0 = torch.unsqueeze(origin_ten[useit0], 0)
                origin_samples = origin_ten0
                origin_new_targets.append(origin_targets[useit0])
                origin_targets = [{k: v for k, v in t.items()} for t in origin_new_targets]
                return samples, targets, origin_samples, origin_targets, True
            
            #origin_samples, _ = origin_samples.decompose()
            return samples, targets, origin_samples, origin_targets, True
            
        if batch_size == 3 :
            if no_use == 3:
                return samples, targets, origin_samples, origin_targets, False
            if no_use == 2:
                new_targets = []
                origin_new_targets = []
                useit0 = yes_use[0]
                ten, mask = samples.decompose()
                
                ten0 = torch.unsqueeze(ten[useit0], 0)
                mask0 = torch.unsqueeze(mask[useit0], 0)
                
                samples = utils.NestedTensor(ten0, mask0)
                new_targets.append(targets[useit0])
                targets = [{k: v for k, v in t.items()} for t in new_targets]
                return samples, targets, origin_samples, origin_targets, True
            
            if no_use == 1:
                new_targets = []
                origin_new_targets = []
                useit0 = yes_use[0]
                useit1 = yes_use[1]
                ten, mask = samples.decompose()
                
                ten0 = torch.unsqueeze(ten[useit0], 0)
                mask0 = torch.unsqueeze(mask[useit0], 0)
                ten1 = torch.unsqueeze(ten[useit1], 0)
                mask1 = torch.unsqueeze(mask[useit1], 0)
                
                ten = torch.cat([ten0,ten1], dim = 0) 
                mask = torch.cat([mask0,mask1], dim = 0) 
                
                samples = utils.NestedTensor(ten, mask)
                #origin_samples = utils.NestedTensor(origin_ten, origin_mask)
                
                new_targets.append(targets[useit0])
                new_targets.append(targets[useit1])
                targets = [{k: v for k, v in t.items()} for t in new_targets]
                
                return samples, targets, origin_samples, origin_targets, True 
            
            return samples, targets, origin_samples, origin_targets, True #return original
        
        if batch_size == 4 :
            if no_use == 4:
                return samples, targets, origin_samples, origin_targets, False
            if no_use == 3:
                new_targets = []
                useit0 = yes_use[0]
                ten, mask = samples.decompose()
                
                ten0 = torch.unsqueeze(ten[useit0], 0)
                mask0 = torch.unsqueeze(mask[useit0], 0)
                
                samples = utils.NestedTensor(ten0, mask0)
                new_targets.append(targets[useit0])
                targets = [{k: v for k, v in t.items()} for t in new_targets]
                return samples, targets, origin_samples, origin_targets, True
            if no_use == 2:
                new_targets = []
                origin_new_targets = []
                useit0 = yes_use[0]
                useit1 = yes_use[1]
                ten, mask = samples.decompose()
                
                ten0 = torch.unsqueeze(ten[useit0], 0)
                mask0 = torch.unsqueeze(mask[useit0], 0)
                ten1 = torch.unsqueeze(ten[useit1], 0)
                mask1 = torch.unsqueeze(mask[useit1], 0)
                
                ten = torch.cat([ten0,ten1], dim = 0) 
                mask = torch.cat([mask0,mask1], dim = 0) 
                
                samples = utils.NestedTensor(ten, mask)
                
                new_targets.append(targets[useit0])
                new_targets.append(targets[useit1])
                targets = [{k: v for k, v in t.items()} for t in new_targets]
                
                new_targets = []
                useit0 = yes_use[0]
                useit1 = yes_use[1]
                useit2 = yes_use[2]
                ten, mask = samples.decompose()
                
                ten0 = torch.unsqueeze(ten[useit0], 0)
                mask0 = torch.unsqueeze(mask[useit0], 0)
                ten1 = torch.unsqueeze(ten[useit1], 0)
                mask1 = torch.unsqueeze(mask[useit1], 0)
                ten2 = torch.unsqueeze(ten[useit2], 0)
                mask2 = torch.unsqueeze(mask[useit2], 0)

                ten = torch.cat([ten0,ten1, ten2], dim = 0) 
                mask = torch.cat([mask0,mask1, mask2], dim = 0) 
                
                samples = utils.NestedTensor(ten, mask)
                
                new_targets.append(targets[useit0])
                new_targets.append(targets[useit1])
                new_targets.append(targets[useit2])
                targets = [{k: v for k, v in t.items()} for t in new_targets]
                return samples, targets, origin_samples, origin_targets, True 
            
            return samples, targets, origin_samples, origin_targets, True #return original
        
        raise Exception("set your batch_size : 2 or 3")
    return wrapper

# how many classes in targets for couning each isntances 
def check_class(verbose: bool, targets: Dict, label_dict: Dict, 
                DID_COUNT: int =4000, VE_COUNT: int=7000, PZ_COUNT: int=4000, CL_Limited: int = 0):
    

    limit2 = [28, 32, 35, 41, 56] #photozone
    limit3 = [22, 23, 24, 25, 26, 27, 29, 31, 33, 37, 39, 40, 45, 46, 48, 49, 51, 52, 58, 59] #VE 
    no_use = []
    yes_use = []
    check_list = []
    
        
    for enum, target in enumerate(targets):
        
        label_tensor = target['labels']
        label_tensor = label_tensor.cpu()
        label_tensor_unique = torch.unique(label_tensor)
        
        # Normal Limited Training (for equivalent Performance)
        if CL_Limited is 0 :
            check_list = [idx.item() for idx in label_tensor_unique if idx.item() in label_dict  if idx.item() <= 21 and label_dict[idx.item()] > DID_COUNT] #did
            check_list2 = [idx.item() for idx in label_tensor_unique if idx.item() in label_dict  if idx.item() in limit2 and label_dict[idx.item()] > PZ_COUNT] #pz
            check_list3 = [idx.item() for idx in label_tensor_unique if idx.item() in label_dict  if idx.item() in limit3 and label_dict[idx.item()] > VE_COUNT] #ve (more)
            check_list4 = []
        else:
            # Continual Limited Training (for equivalent Performance)
            check_list = [idx.item() for idx in label_tensor_unique if idx.item() in label_dict  if idx.item() <= 21 and label_dict[idx.item()] > CL_Limited] #did
            check_list2 = [idx.item() for idx in label_tensor_unique if idx.item() in label_dict  if idx.item() in limit2 and label_dict[idx.item()] > CL_Limited] #pz
            check_list3 = [idx.item() for idx in label_tensor_unique if idx.item() in label_dict  if idx.item() in limit3 and label_dict[idx.item()] > CL_Limited] #ve (more)
            check_list4 = [idx.item() for idx in label_tensor_unique if idx.item() in label_dict  if label_dict[idx.item()] > CL_Limited] #ve (other)
            
        #TODO : before checking overlist Process outputdim and continue iter
        if len(check_list) > 0 or len(check_list2) > 0 or len(check_list3) > 0 or len(check_list4) > 0:
            if utils.is_main_process() and verbose:
                print("overlist: ", check_list, check_list2, check_list3, check_list4)
            no_use.append(enum)

        else:
            yes_use.append(enum)
            label_tensor_count = label_tensor.numpy()
            bin = np.bincount(label_tensor_count)
            for idx in label_tensor_unique:
                idx = idx.item()
                if idx in label_dict.keys():
                    label_dict[idx] += bin[idx]
                else :
                    label_dict[idx] = bin[idx]

    return no_use, yes_use, label_dict

from torch.utils.data import DataLoader
from datasets import build_dataset, get_coco_api_from_dataset
def new_dataLoader(saved_dict, args):
    #print(f"{dist.get_rank()}gpu training saved dict : {saved_dict}")
    dataset_idx_list = []
    for _, value in saved_dict.items():
        if len(value) > 0 :
            np_idx_list = np.array(value, dtype=object)
            dataset_idx_list.extend(np.unique(np_idx_list[:, 3]).astype(np.uint8).tolist())
    #print(f"{dist.get_rank()} gpu dataset_idx_list : {dataset_idx_list}")
    
    custom_dataset = build_dataset(image_set='train', args=args, img_ids=dataset_idx_list)
    
    custom_loader = DataLoader(custom_dataset, args.batch_size, shuffle=True, collate_fn=utils.collate_fn, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    return custom_loader


def _dataset_for_memory_check(*args):

    check_sample = args[2]
    check_target = args[3]
    
    return check_sample, check_target

def _divede_targetset(target: Dict, index: int)-> Dict:
    
    changed_target_boxes = target['boxes'][target["labels"] == index]
    changed_target_area = target['area'][target["labels"] == index]
    changed_target_iscrowd = target['iscrowd'][target["labels"] == index]
    changed_target_labels = target['labels'][target["labels"] == index]
    
    return {"boxes" : changed_target_boxes, "labels" : changed_target_labels, "image_id" : target["image_id"], "area" : changed_target_area,
            "iscrowd" : changed_target_iscrowd, "orig_size" : target["orig_size"], "size" : target["size"]}

@decompose #For Divide Dataset
def _rearrange_targets(no_use_count: int, samples: utils.NestedTensor, targets: Dict, origin_samples: utils.NestedTensor, origin_targets: Dict, 
                      used_number: List) -> Tuple[int, List, utils.NestedTensor, Dict, utils.NestedTensor, Dict, List]:
    no_use_count = 1 
    batch_size = 2
    return (batch_size, no_use_count, samples, targets, origin_samples, origin_targets, used_number)

from pympler import asizeof, summary
def contruct_rehearsal(losses_value: float, lower_limit: float, upper_limit: float, samples, targets,
                       origin_samples: torch.Tensor, origin_targets: Dict, rehearsal_classes: Dict, Current_Classes: List[int], Rehearsal_Memory: int = 300) \
                           -> Dict:
    # Check if losses_value is within the specified range
    if losses_value > lower_limit and losses_value < upper_limit : 
        ex_device = torch.device("cpu")
        
        for enum, target in enumerate(targets):
            # Get the unique labels and the count of each label
            label_tensor = target['labels']
            label_tensor_unique = torch.unique(label_tensor)
            if set(label_tensor_unique.tolist()).issubset(Current_Classes) is False: #if unique tensor composed by Old Dataset, So then Continue iteration
                continue
            
            label_tensor_count = label_tensor.numpy()
            bin = np.bincount(label_tensor_count)
            
            # Get origin sample and target that devided to each instance
            _, _, new_sample, new_target, _ = _rearrange_targets(no_use_count=None, samples=samples, targets=targets, origin_samples= origin_samples, origin_targets= origin_targets,
                               used_number=[enum])

            #Rehearsal. 오름차순 정렬하고, Loss가 큰 값들부터 제거 -> Loss가 크면 대표성이 떨어짐.
            for unique_idx in label_tensor_unique:
                unique_idx = unique_idx.item()
                divided_target = _divede_targetset(new_target[0], int(unique_idx))
                
                if unique_idx in rehearsal_classes.keys():  # Check if the unique label already exists in the rehearsal_classes dictionary
                    rehearsal_classes[unique_idx] = sorted(rehearsal_classes[unique_idx], key = lambda x : x[1]) # ASC by loss 
                    for_usage_check_list = [(_dataset_for_memory_check(*values)) for values in rehearsal_classes[unique_idx] if _dataset_for_memory_check(*values) is not None]
                    # check unique_idx samples capacity
                    instances_bytes = asizeof.asizeof(for_usage_check_list) 
                    memory_usage_MB = instances_bytes * 0.00000095367432
                    
                    if memory_usage_MB <= Rehearsal_Memory: # construct based on capacity / # If the memory usage is greater than 500MB, replace the sample with the highest loss value with the new sample
                        rehearsal_classes[unique_idx].append([bin[unique_idx], losses_value, new_sample, divided_target])
                    else :
                        if rehearsal_classes[unique_idx][-1][1] > losses_value:
                            rehearsal_classes[unique_idx][-1] = [bin[unique_idx], losses_value, new_sample, divided_target]
                else : 
                    rehearsal_classes[unique_idx] = [[bin[unique_idx], losses_value, new_sample, divided_target]]
        
    return rehearsal_classes

def rearrange_rehearsal(rehearsal_classes: dict, current_classes: list) -> dict:
    '''
        delete bincount, loss_value in Current class 
    '''
    for key, contents in rehearsal_classes.items():
        if key in current_classes:
            for content in contents:
                del content[:2]
            
    return rehearsal_classes

def load_model_params(model: model,
                      dir: str = "/data/LG/real_dataset/total_dataset/test_dir/Deformable-DETR/loss_2_3/checkpoint0327.pth"):
    new_model_dict = model.state_dict()
    #temp dir
    checkpoint = torch.load(dir)
    pretraind_model = checkpoint["model"]
    name_list = [name for name in new_model_dict.keys() if name in pretraind_model.keys()]
    name_list = list(filter(lambda x : "class" not in x, name_list))
    pretraind_model_dict = {k : v for k, v in pretraind_model.items() if k in name_list } # if "class" not in k => this method used in diff class list
    
    new_model_dict.update(pretraind_model_dict)
    model.load_state_dict(new_model_dict)


    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    #No parameter update
    for name, params in model.named_parameters():
        if name in pretraind_model_dict.keys():
            params.requires_grad = True #if you wanna set frozen the pre parameters for specific Neuron update, so then you could set False
        else:
            params.requires_grad = True
            
            
    return model


def save_model_params(model_without_ddp:model, optimizer:torch.optim, lr_scheduler:scheduler,
                     args:arg, output_dir: str, TASK_num:int, Total_task:int):
    '''
        Save the model for each task
    '''
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        print(f"Directroy created")
        
    checkpoint_paths = output_dir + f'checkpoint{TASK_num:02}_{Total_task:02}.pth'
    utils.save_on_master({
        'model': model_without_ddp.state_dict(), 
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'args': args,
    }, checkpoint_paths)
    