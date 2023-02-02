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
    def __init__(self, loader, device, prefetch=True, Mosaic=False):
        self.loader = iter(loader)
        self.prefetch = prefetch
        self.device = device
        self.Mosaic = Mosaic
        if prefetch:
            self.stream = torch.cuda.Stream()
            self.preload()

    def preload(self):
        try:
            if self.Mosaic == True:
                self.next_samples, self.next_targets, self.next_origin_samples, self.next_origin_targets, self.next_Current_samples, self.next_Current_target, self.next_Diff_samples, self.next_Diff_targets= next(self.loader)
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
                self.next_Current_samples, self.next_Current_target = to_cuda(self.next_Current_samples, self.next_Current_target, self.device)
                self.next_Diff_samples, self.next_Diff_samples = to_cuda(self.next_Diff_samples, self.next_Diff_samples, self.device)
            else:
                self.next_samples, self.next_targets = to_cuda(self.next_samples, self.next_targets, self.device)

    def next(self):
        if self.Mosaic:
            if self.prefetch:
                torch.cuda.current_stream().wait_stream(self.stream)
                samples = self.next_samples
                targets = self.next_targets
                origin_samples = self.next_origin_samples
                origin_targets = self.next_origin_targets
                current_samples = self.next_Current_samples
                current_targets = self.next_Current_targets
                Diff_samples = self.next_Diff_samples
                Diff_targets = self.next_Diff_targets
                
                if samples is not None:
                    samples.record_stream(torch.cuda.current_stream())
                if targets is not None:
                    for t in targets:
                        for k, v in t.items():
                            v.record_stream(torch.cuda.current_stream())
                            
                if current_samples is not None:
                    current_samples.record_stream(torch.cuda.current_stream())
                if current_targets is not None:
                    for t in current_targets:
                        for k, v in t.items():
                            v.record_stream(torch.cuda.current_stream())
                            
                if Diff_samples is not None:
                    Diff_samples.record_stream(torch.cuda.current_stream())
                if Diff_targets is not None:
                    for t in Diff_targets:
                        for k, v in t.items():
                            v.record_stream(torch.cuda.current_stream())
                self.preload()
            else:
                try:
                    samples, targets, origin_samples, origin_targets, current_samples, current_targets, Diff_samples, Diff_targets = next(self.loader)
                    samples, targets = to_cuda(samples, targets, self.device)
                    current_samples, current_targets = to_cuda(samples, targets, self.device)
                    Diff_samples, Diff_targets = to_cuda(samples, targets, self.device)
                except StopIteration:
                    samples = None
                    targets = None
            return samples, targets, origin_samples, origin_targets, current_samples, current_targets, Diff_samples, Diff_targets

        else:
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
                self.preload()
            else:
                try:
                    samples, targets, origin_samples, origin_targets = next(self.loader)
                    samples, targets = to_cuda(samples, targets, self.device)
                except StopIteration:
                    samples = None
                    targets = None
            return samples, targets, origin_samples, origin_targets
