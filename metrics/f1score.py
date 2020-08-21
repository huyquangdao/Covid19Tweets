from sklearn.metrics import f1_score
from metrics.metric import Metric

class F1Score(Metric):

    def __init__(self, average_type = 'micro', pos_label = 1):
        super().__init__()
        self.result = []
        self.pos_label = pos_label
        self.average_type = average_type
    
    def step(self, y_true, y_pred):
        batch_result =  f1_score(y_true, y_pred, average = self.average_type, pos_label=self.pos_label)
        self.result.append(batch_result)
        return batch_result
    
    def execute(self):
        return sum(self.result) / len(self.result)
    
    def reset(self):
        self.result = []