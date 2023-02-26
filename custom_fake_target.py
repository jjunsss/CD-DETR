import torch
import torch.nn.functional as F
from torch import nn
import math

from util import box_ops


def normal_query_selc_to_target(outputs, targets, current_classes):
    out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
    
    prob = out_logits.sigmoid()
    topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 30, dim=1)
    scores = topk_values
    topk_boxes = topk_indexes // out_logits.shape[2]
    labels = topk_indexes % out_logits.shape[2]
    boxes = torch.gather(out_bbox, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))

    results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
    threshold = 0.5
    current_classes = min(current_classes)
    for target, result in zip(targets, results):
        if target["labels"][target["labels"] < current_classes].shape[0] > 0: #Old Class에서만 동작하도록 구성
            continue
        
        scores = result["scores"][result["scores"] > threshold]
        labels = result["labels"][result["scores"] > threshold] 
        boxes = result["boxes"][result["scores"] > threshold]

        
        if labels[labels < current_classes].size(0) > 0:
            addlabels = labels[labels < current_classes]
            addboxes = boxes[labels < current_classes]
            area = addboxes[:, 2] * addboxes[:, 3]
            addboxes += 1e-10
            print("old fake query operation")
            target["boxes"] = torch.cat((target["boxes"], addboxes))
            target["labels"] = torch.cat((target["labels"], addlabels))
            #target["area"] = torch.cat((target["area"], area))
            #target["iscrowd"] = torch.cat((target["iscrowd"], torch.tensor([0], device = torch.device("cuda"))))
        
    return targets

def only_oldset_mosaic_query_selc_to_target(outputs, targets, current_classes):
    out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
    
    prob = out_logits.sigmoid()
    topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 30, dim=1)
    scores = topk_values
    topk_boxes = topk_indexes // out_logits.shape[2]
    labels = topk_indexes % out_logits.shape[2]
    boxes = torch.gather(out_bbox, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))

    results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
    threshold = 0.5
    current_classes = min(current_classes)
    for target, result in zip(targets, results):
        if target["labels"][target["labels"] >= current_classes].shape[0] > 0: #New Class에서만 동작하도록 구성
            continue
        
        scores = result["scores"][result["scores"] > threshold]
        labels = result["labels"][result["scores"] > threshold] 
        boxes = result["boxes"][result["scores"] > threshold]

        
        if labels[labels >= current_classes].size(0) > 0:
            addlabels = labels[labels >= current_classes]
            addboxes = boxes[labels >= current_classes]
            area = addboxes[:, 2] * addboxes[:, 3]
            addboxes += 1e-10
            print("new fake query operation")
            target["boxes"] = torch.cat((target["boxes"], addboxes))
            target["labels"] = torch.cat((target["labels"], addlabels))
            #target["area"] = torch.cat((target["area"], area))
            #target["iscrowd"] = torch.cat((target["iscrowd"], torch.tensor([0], device = torch.device("cuda"))))
        
    return targets

def mosaic_query_selc_to_target(outputs, targets, current_classes):
    out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

    prob = out_logits.sigmoid()
    topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 30, dim=1)
    scores = topk_values
    topk_boxes = topk_indexes // out_logits.shape[2]
    labels = topk_indexes % out_logits.shape[2]
    boxes = torch.gather(out_bbox, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))

    results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
    threshold = 0.5
    current_classes = max(current_classes)
    for target, result in zip(targets, results):
        scores = result["scores"][result["scores"] > threshold]
        labels = result["labels"][result["scores"] > threshold] 
        boxes = result["boxes"][result["scores"] > threshold]
        
        if (labels < current_classes).sum() == 0: # check the shape of labels before indexing boxes
            continue
        addlabels = labels[labels < current_classes]
        addboxes = boxes[labels < current_classes]
        
        count_prebox = addboxes.shape[0]
        count_target = target["boxes"].shape[0]
        # create coordinate grids from the tensors
        for src_idx in range(count_prebox):
            overlapped = False
            for tgt_idx in range(count_target):
                iou, _ = box_ops.box_iou(
                    box_ops.box_cxcywh_to_xyxy(addboxes[src_idx].unsqueeze(0)),
                    box_ops.box_cxcywh_to_xyxy(target["boxes"][tgt_idx].unsqueeze(0)))
                
                if iou > 0.0 :
                    overlapped = True
                    break
            
            if overlapped == False:    
                area = boxes[src_idx, 2] * boxes[src_idx, 3]
                addboxes[src_idx] += 1e-10
                target["boxes"] = torch.cat((target["boxes"], addboxes[src_idx].unsqueeze(0)))
                target["labels"] = torch.cat((target["labels"], addlabels[src_idx].unsqueeze(0)))
                print(f"Mosaic fake query operation")
                #target["area"] = torch.cat((target["area"], area.unsqueeze(0)))
                #target["iscrowd"] = torch.cat((target["iscrowd"], torch.tensor([0], device = torch.device("cuda"))))

    return targets