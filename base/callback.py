
class CallBack(object):

    def __init__(self):
        pass

    def step(self):
        raise NotImplementedError('You must implement this method')

    def execute(self):
        raise NotImplementedError('You must implement this method')
