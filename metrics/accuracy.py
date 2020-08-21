from sklearn.metrics import accuracy_score
from metrics.metric import Metric

class AccuracyScore(Metric):

    def __init__(self):
        super().__init__()
        self.result = []
    
    def step(self, y_true, y_pred):
        batch_result =  accuracy_score(y_true, y_pred)
        self.result.append(batch_result)
        return batch_result
    
    def execute(self):
        return sum(self.result) / len(self.result)
    
    def reset(self):
        self.result = []