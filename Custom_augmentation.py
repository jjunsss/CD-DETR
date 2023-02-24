import torch
import numpy as np
from typing import Dict, List
import random
import matplotlib.pyplot as plt
import albumentations as A
import cv2
import copy
import matplotlib.pyplot as plt
from util.box_ops import box_cxcywh_to_xyxy_resize, box_xyxy_to_cxcywh
import datasets.transforms as T
import copy
from datasets.coco import origin_transform

def visualize_bboxes(img, bboxes, img_size = (1024, 1024), vertical = False):
    # min_or = img.min()
    # max_or = img.max()
    # img_uint = ((img - min_or) / (max_or - min_or) * 255.).astype(np.uint8)
    plt.figure(figsize=(10, 10))
    plt.axis("off")
    h, w = img_size
    img = img[..., ::-1]
    if vertical == False:
        for bbox in bboxes:
            xmin, ymin, xmax, ymax, cls = bbox * torch.tensor((w , h, w, h, 1))
            cv2.rectangle(img,(int(xmin.item()), int(ymin.item())),(int(xmax.item()), int(ymax.item())), (255, 0, 0), 3)
            label = f'Class {cls.item()}'
            cv2.putText(img, label, (int(xmin.item()), int(ymin.item()) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
        cv2.imwrite("./Combined_"+str(bboxes[0][-1])+".png",img)
    else:
        #bboxes = bboxes['boxes']
        bboxes = box_cxcywh_to_xyxy_resize(bboxes)
        x1, y1, x2, y2 = bboxes.unbind(-1)
        bboxes = torch.stack([x1, y1, x2, y2 ], dim=-1).tolist()
        for bbox in bboxes:
            xmin, ymin, xmax, ymax = torch.tensor(bbox) * torch.tensor((w , h, w, h))
            cv2.rectangle(img,(int(xmin.item()), int(ymin.item())),(int(xmax.item()), int(ymax.item())), (255, 0, 0), 3)
        cv2.imwrite("./vertical_"+str(bboxes[0][-1])+".png",img)
        
import torchvision.transforms.functional as F
class CCB(object):
    def __init__(self, image_size, Continual_Batch = 2):
        self.img_size = image_size
        self.transformed = origin_transform("custom")
        self.Continual_Batch = Continual_Batch
        
    def __call__(self, image_list, target_list):
        if self.Continual_Batch == 3:
            Cur_img, Cur_lab, Dif_img, Dif_lab = self._load_mosaic(image_list, target_list)#np.array(original) / norm coord torch.tensor / np.array(original) / norm coord torch.tensor
            Cur_img, Cur_lab = self.transformed(Cur_img, Cur_lab) #Adapt Normalization(Nomalized image and ToTensor)
            Dif_img, Dif_lab = self.transformed(Dif_img, Dif_lab) #Adapt Normalization(Nomalized image and ToTensor)
            #visualize_bboxes(np.clip(Dif_img.permute(1, 2, 0).numpy(), 0, 1).copy(), Dif_lab['boxes'], self.img_size, True)
            return Cur_img, Cur_lab, Dif_img, Dif_lab
        
        if self.Continual_Batch == 2:
            Cur_img, Cur_lab, _, _ = self._load_mosaic(image_list, target_list)
            Cur_img, Cur_lab = self.transformed(Cur_img, Cur_lab)
            #visualize_bboxes(np.clip(Cur_img.permute(1, 2, 0).numpy(), 0, 1).copy(), Cur_lab['boxes'], Cur_img.shape[:-1], True)
            return Cur_img, Cur_lab
    
   
    def _load_mosaic(self, image_list, target_list):
        '''
            Current_mosaic_index : For constructing masaic about current classes
            Diff_mosaic_index : For constructing mosaic abhout differenct classes (Not Now classes)
            Current_bboxes : numpy array. [cls, cx, cy, w, h] for current classes
            Diff_bboxes : numpy array. [cls, cx, cy, w, h] for different classes (Not Now classes)
        '''
        # loads images in a mosaic
        Mosaic_size = self.img_size #1024, im_w, im_h : 1024
            
        Current_mosaic_img, Current_mosaic_labels = self._make_batch_mosaic(image_list, target_list, Mosaic_size)
        Current_mosaic_labels = self._make_resized_targets(Current_mosaic_labels)
        
        if self.Continual_Batch == 3: #For 3 CBB Training
            Diff_mosaic_labels = copy.deepcopy(Current_mosaic_labels)
            Diff_mosaic_img, Diff_bbox, Diff_labels  = _HorizontalFlip(Current_mosaic_img, Current_mosaic_labels['boxes'], Current_mosaic_labels['labels'])
            Diff_mosaic_labels = self._make_resized_targets(Diff_bbox, Diff_labels)
            return Current_mosaic_img, Current_mosaic_labels, Diff_mosaic_img, Diff_mosaic_labels
        return Current_mosaic_img, Current_mosaic_labels, None, None #For 2 CBB Training

    def _make_batch_mosaic(self, image_list, target_list, mosaic_size ):
        mosaic_aug_labels = []
        for i, (img, target) in enumerate(zip(image_list, target_list)):
            # Load image
            transposed_img, transposed_bboxes = self._augment_bboxes(img, target) #! cv2.imread 통해서 불러옴. 나는 coco 사용하기에 변경해야 함.
            height, width, channel, = transposed_img.shape #H W C(Numpy Img)
            temp_bbox = transposed_bboxes.clone().detach()
            temp_bbox[:, :-1] /= 2
            # place img in img4(특정 center point 잡아서 할당)
            if i == 0:  # top left
                mosaic_aug_img = np.full((mosaic_size[0], mosaic_size[1], channel), 114, dtype=np.uint8)  # base image with 4 tiles
                mosaic_aug_img[:height, :width, :] = transposed_img
                mosaic_bboxes = temp_bbox.clone().detach()
                continue
            elif i == 1:  # top right
                mosaic_aug_img[:height, width:, :] = transposed_img
                temp_bbox[:, 0] += 0.5
                temp_bbox[:, 2] += 0.5
            elif i == 2:  # bottom left
                mosaic_aug_img[height:, :width, :] = transposed_img
                temp_bbox[:, 1] += 0.5
                temp_bbox[:, 3] += 0.5
            elif i == 3:  # bottom right
                mosaic_aug_img[height:, width:, :] = transposed_img
                temp_bbox[:, :-1] += 0.5
                
            mosaic_bboxes = torch.vstack((temp_bbox, mosaic_bboxes))
        #visualize_bboxes(mosaic_aug_img, mosaic_bboxes, self.img_size)
        return mosaic_aug_img, mosaic_bboxes
    
    def _augment_bboxes(self, img:np.array, target:torch.tensor): #* Checking
        '''
            maybe index_list is constant shape in clockwise(1:origin / 2:Current Img / 3: Currnt image / 4: Current img)
        '''
        boxes = target["boxes"] #* Torch tensor
        classes = target["labels"]
        
        boxes = box_cxcywh_to_xyxy_resize(boxes)
        x1, y1, x2, y2 = boxes.unbind(-1)
        bboxes = torch.stack([x1, y1, x2, y2, classes.long()], dim=-1)
        #bboxes = torch.stack([x1, y1, x2, y2, classes.long()], dim=-1).tolist()

        transposed_img, transposed_bboxesd = self._Resize_for_batchmosaic(img, int(self.img_size[0]/2), int(self.img_size[1]/2), bboxes)
        
        return transposed_img, transposed_bboxesd
    
    def _Resize_for_batchmosaic(self, img:np.array, height_resized:int , width_resized:int , bboxes:torch.tensor): #* Checking
        """
            img : torch.tensor(Dataset[idx])
            resized : size for resizing
            BBoxes : resize image to (height, width) in a image
        """
        #이미지 변환 + Box Label 변환
        temp_img = copy.deepcopy(img)
        bboxes[:, :-1].clamp_(min = 0.0, max=1.0)
        bboxes.tolist()
        
        transform = A.Compose([
            A.Resize(height_resized, width_resized)
        ], bbox_params=A.BboxParams(format='albumentations'))

        #Annoation change
        transformed = transform(image = temp_img, bboxes = bboxes)
        transformed_bboxes = transformed['bboxes']
        transformed_img = transformed["image"]
        #visualize_bboxes(transformed_img, transformed_bboxes)
        
        transformed_bboxes = torch.tensor(transformed_bboxes)
        #transformed_img = torch.tensor(transformed_img, dtype=torch.float32).permute(2, 0, 1) #TODO: change dimension permunate for training in torch image
        
        return  transformed_img, transformed_bboxes 

    def _make_resized_targets(self, target: Dict, v_labels: torch.tensor = None)-> Dict:
        
        temp_dict = {}
        if v_labels is not None :
            boxes = target
            labels = v_labels
        else:
            boxes = target[:, :-1]
            labels = target[:, -1]
        cxcy_boxes = box_xyxy_to_cxcywh(boxes)
        temp_dict['boxes'] = cxcy_boxes.to(dtype=torch.float32)
        temp_dict['labels'] = labels.to(dtype=torch.long)
        temp_dict['images_id'] = torch.tensor(0)
        temp_dict['area'] = torch.tensor(0)
        temp_dict['iscrowd'] = torch.tensor(0)
        temp_dict['orig_size'] = torch.tensor(self.img_size)
        temp_dict['size'] = torch.tensor(self.img_size)
        
        return temp_dict
    
def _HorizontalFlip(img:np.array, bboxes, labels): #* Checking
    """
        img : torch.tensor(Dataset[idx])
        resized : size for resizing
        BBoxes : resize image to (height, width) in a image
    """
 
    boxes = box_cxcywh_to_xyxy_resize(bboxes)
    boxes[:, :-1].clamp_(min = 0.0, max=1.0)
    x1, y1, x2, y2 = boxes.unbind(-1)
    
    boxes = torch.stack([x1, y1, x2, y2, labels], dim=-1).tolist()
    
    # bboxes = bboxes.tolist()
    class_labels = labels.tolist()
    temp_img = copy.deepcopy(img)
    
    transform = A.Compose([
        A.HorizontalFlip(1),
        #A.VerticalFlip(0.1),
    ], bbox_params=A.BboxParams(format='albumentations'))
    
    #Annoation change
    transformed = transform(image = temp_img, bboxes = boxes)
    transformed_bboxes = transformed['bboxes']
    transformed_img = transformed["image"]
    
    temp = torch.tensor(transformed_bboxes)
    transformed_bboxes = temp[:, :-1]
    transformed_labels = temp[:, -1]
    transformed_img = torch.tensor(transformed_img, dtype=torch.float32).permute(2, 0, 1)
    #visualize_bboxes(np.clip(transformed_img.permute(1, 2, 0).numpy(), 0, 1).copy(), transformed_bboxes, (1024, 1024), True)
    return  transformed_img, transformed_bboxes, transformed_labels 