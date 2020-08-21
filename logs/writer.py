import os
import pandas as pd
from base.callback import CallBack

class Writer(CallBack):

    def __init__(self):
        super(Writer, self)
        self.memory = {}
        self.data_frame = None
    
    def step(self, results, field_name, iterations):
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
        temp_dataframe = pd.DataFrame(result, index=[0])
        if self.data_frame is not None:
            self.data_frame = pd.concat([self.data_frame, temp_dataframe],axis=0)
        else:
            self.data_frame = temp_dataframe
        print(self.data_frame)
    
    def reset(self):
        self.memory = {}