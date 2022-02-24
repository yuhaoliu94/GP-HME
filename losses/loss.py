class Loss(object):

    def __init__(self, dout):
        self.dout = dout

    def eval(self, _ytrue, _ypred):
        raise NotImplementedError("Subclass should implement this.")

    def get_name(self):
        raise NotImplementedError("Subclass should implement this.")
