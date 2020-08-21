from base.callback import CallBack

class Metric(CallBack):

    def __init__(self):
        super(Metric, self).__init__()
    
    def step(self):
        raise NotImplementedError('You must implement this method')

    def excecute(self):
        raise NotImplementedError('You must implement this method')
