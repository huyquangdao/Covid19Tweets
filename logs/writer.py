import os
import pandas as pd

from torch.utils.tensorboard import SummaryWriter
from base.callback import CallBack


class Writer(CallBack):

    def __init__(self):
        pass
    
    def step(self):
        pass

    def execute(self):
        pass
    
    def reset(self):
        pass

class PandasWriter(Writer):

    def __init__(self):
        super(PandasWriter, self).__init__()
        self.memory = {}
        self.data_frame = None
        self.all_memory = {}
    
    def step(self, results, global_step = None):
        for key, item in results.items():
            if not key in self.memory:
                self.memory[key] = [item]
            else:
                self.memory[key].append(item)
    
    def execute(self, epoch):
        result = {}
        result['epoch'] = epoch
        for key, item in self.memory.items():
            if isinstance(item, list):
                result[key] = sum(item) / len(item)
            if key not in self.all_memory:
                self.all_memory[key] = self.memory[key]
            else:
                self.all_memory[key].extend(self.memory[key])
        temp_dataframe = pd.DataFrame(result, index=[0])
        if self.data_frame is not None:
            self.data_frame = pd.concat([self.data_frame, temp_dataframe],axis=0)
        else:
            self.data_frame = temp_dataframe
        print(self.data_frame)
    
    def reset(self):
        self.memory = {}

class TensorboardWriter(Writer):

    def __init__(self, log_dir = 'logs_dir'):
        super(TensorboardWriter, self).__init__()
        self.writer = SummaryWriter(log_dir=log_dir)

    
    def step(self, results, global_step = None):
        if global_step is None:
            for key, item in results.items():
                self.writer.add_scalar(key, item, global_step)
    
    def execute(self, epoch):
        pass
    