# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import torch

def to_cuda(samples, targets, device):
    samples = samples.to(device, non_blocking=True)
    targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
    return samples, targets

class data_prefetcher():
    def __init__(self, loader, device, prefetch=True, Mosaic=False, Continual_Batch=2):
        self.loader = iter(loader)
        self.prefetch = prefetch
        self.device = device
        self.Mosaic = Mosaic
        self.data_gen = None
        self.Continual_Batch = Continual_Batch
        if prefetch:
            self.stream = torch.cuda.Stream()
            self.preload()

    def preload(self, new = False):
        try:
            if self.Mosaic == True:
                if self.data_gen is None:
                    if self.Continual_Batch == 3:
                        self.next_samples, self.next_targets, self.next_origin_samples, self.next_origin_targets, self.next_Current_samples, self.next_Current_target, self.next_Diff_samples, self.next_Diff_targets= next(self.loader)
                        temp = [[self.next_samples, self.next_targets ,self.next_origin_samples, self.next_origin_targets], [self.next_Current_samples, self.next_Current_target, None, None], [self.next_Diff_samples, self.next_Diff_targets, None, None]]
                    else: #CB = 2
                        self.next_samples, self.next_targets, self.next_origin_samples, self.next_origin_targets, self.next_Current_samples, self.next_Current_target = next(self.loader)
                        temp = [[self.next_samples, self.next_targets ,self.next_origin_samples, self.next_origin_targets], [self.next_Current_samples, self.next_Current_target, None, None]]
                        
                    self.data_gen = self._split_gpu_preload(temp)
                    self.next_samples, self.next_targets, self.next_origin_samples, self.next_origin_targets = next(self.data_gen)
                elif self.data_gen is not None:
                    self.next_samples, self.next_targets, self.next_origin_samples, self.next_origin_targets = next(self.data_gen, (None, None, None, None))
                    
                    if self.next_samples is None or new == True:
                        if self.Continual_Batch == 3:
                            self.next_samples, self.next_targets, self.next_origin_samples, self.next_origin_targets, self.next_Current_samples, self.next_Current_target, self.next_Diff_samples, self.next_Diff_targets= next(self.loader)
                            temp = [[self.next_samples, self.next_targets ,self.next_origin_samples, self.next_origin_targets], [self.next_Current_samples, self.next_Current_target, None, None], [self.next_Diff_samples, self.next_Diff_targets, None, None]]
                        else: #CB = 2
                            self.next_samples, self.next_targets, self.next_origin_samples, self.next_origin_targets, self.next_Current_samples, self.next_Current_target = next(self.loader)
                            temp = [[self.next_samples, self.next_targets ,self.next_origin_samples, self.next_origin_targets], [self.next_Current_samples, self.next_Current_target, None, None]]
                        self.data_gen = self._split_gpu_preload(temp)
                        self.next_samples, self.next_targets, self.next_origin_samples, self.next_origin_targets = next(self.data_gen)
            else:
                self.next_samples, self.next_targets, self.next_origin_samples, self.next_origin_targets = next(self.loader)
                
        except StopIteration:
            self.next_samples = None
            self.next_targets = None
            self.next_origin_samples = None
            self.next_origin_targets = None
            self.next_Current_samples = None
            self.next_Current_targets = None
            self.next_Diff_samples = None
            self.next_Diff_targets = None
            return
        
        except KeyError:
            self.next_samples = None
            self.next_targets = None
            self.next_origin_samples = None
            self.next_origin_targets = None
            self.next_Current_samples = None
            self.next_Current_targets = None
            self.next_Diff_samples = None
            self.next_Diff_targets = None
            return
        
        with torch.cuda.stream(self.stream):
            if self.Mosaic == True:
                self.next_samples, self.next_targets = to_cuda(self.next_samples, self.next_targets, self.device)
            else:
                self.next_samples, self.next_targets = to_cuda(self.next_samples, self.next_targets, self.device)
    
    def _split_gpu_preload(self, temp):
        for samples, targets, origin_samples, origin_targets in temp:
            if origin_samples is not None :
                yield samples, targets, origin_samples, origin_targets
            else :
                yield samples, targets, None, None #* for Mosaic augmentation Dataset(Current Mosaic, Different Mosaic)
                
    def next(self, new = False):
        if self.prefetch:
            torch.cuda.current_stream().wait_stream(self.stream)
            samples = self.next_samples
            targets = self.next_targets
            origin_samples = self.next_origin_samples
            origin_targets = self.next_origin_targets
            
            if samples is not None:
                samples.record_stream(torch.cuda.current_stream())
            if targets is not None:
                for t in targets:
                    for k, v in t.items():
                        v.record_stream(torch.cuda.current_stream())
            self.preload(new)
        else:
            try:
                samples, targets, origin_samples, origin_targets, current_samples, current_targets, Diff_samples, Diff_targets = next(self.loader)
                samples, targets = to_cuda(samples, targets, self.device)
            except StopIteration:
                samples = None
                targets = None
                
        return samples, targets, origin_samples, origin_targets
