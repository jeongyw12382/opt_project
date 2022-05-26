import numpy as np

class LRSchedulerInterface:

    def __init__(self, lr_init, global_step):
        self.lr_init = lr_init
        self.global_step = global_step

    def get_lr(self, step):
        raise NotImplementedError("")


class ConstLR(LRSchedulerInterface):

    def get_lr(self, step):
        return self.lr_init


class CosineAnnealing(LRSchedulerInterface):

    def get_lr(self, step):
        return self.lr_init * np.cos(step / self.global_step * np.pi / 2)


class LinearWarmupCosineAnnealing(LRSchedulerInterface):

    def get_lr(self, step):
        peak = int(self.global_step * 0.1)
        if step <= peak:
            return self.lr_init * (step / peak)
        else:
            return self.lr_init * np.cos((step - peak) / (self.global_step - peak) * np.pi / 2)


class ExponentialDecay(LRSchedulerInterface):

    def get_lr(self, step):
        t = step / self.global_step
        return np.exp(np.log(self.lr_init) * (1 - t) + np.log(self.lr_init * 0.01) * t)