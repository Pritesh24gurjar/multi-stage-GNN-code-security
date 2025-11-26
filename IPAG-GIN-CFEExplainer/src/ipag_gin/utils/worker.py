class Worker:
    def __init__(self, func):
        self.func = func
    def set_args(self, **args):
        return self.func(**args)
