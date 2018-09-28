import numpy as np

from typing import List, Callable
from torch.nn import Parameter


class LearningGroup(object):
    """
    Responsible for setting learning rates for an arbitrary group of parameters during learning.
    It knows when to unfreeze
    """

    def __init__(self, name: str, parameters: List[Parameter], lr_strategy: Callable,
                 unfreeze_at=0):
        """
        :param name: Name of this learning group.
        :param parameters: Parameters who will share learning rates and frozenness.
        :param lr_strategy: A callable that computes lr based on iter and epoch states.
        :param unfreeze_at: What epoch this parameter should be unfrozen
        """
        self._name = name
        self._lr_strategy = lr_strategy
        self.unfreeze_at = unfreeze_at
        self._parameters = list(parameters)
        self.freeze()

    @property
    def name(self):
        return self._name

    @property
    def parameters(self):
        return self._parameters

    def get_lr(self, epoch, itr):
        return self._lr_strategy(epoch, itr)

    def maybe_unfreeze(self, epoch):
        if epoch >= self.unfreeze_at:
            self.unfreeze()
        return epoch >= self.unfreeze_at

    def freeze(self):
        for p in self._parameters:
            p.requires_grad = False
        self._frozen = True

    def unfreeze(self):
        for p in self._parameters:
            p.requires_grad = True
        self._frozen = False

    def is_frozen(self):
        return self._frozen


class LRScheduler(object):

    def __init__(self, groups: List[LearningGroup], optimizer):
        """
        :param params: Parameters that this scheduler is responsible for updating
        :param optimizer: Optimizer which will be updated with new learning rates
        """
        self._optimizer = optimizer
        self._learning_groups = {gp.name: gp for gp in groups}
        self.epoch = -1
        self.iter = 0
        self.step_epoch()

    def step_epoch(self):
        self.epoch += 1
        self.iter = 0
        for name, lgroup in self._learning_groups.items():
            if lgroup.maybe_unfreeze(self.epoch) and name not in \
                    [p.get('name', '') for p in self._optimizer.param_groups]:
                # this param was just unfrozen, add to optim
                self._optimizer.add_param_group({'name': lgroup.name, 'params': lgroup.parameters})
        # change the learning rates
        self._update_optimizer()

    def step_iter(self):
        self.iter += 1
        self._update_optimizer()

    def _update_optimizer(self):
        """
        Go through and get learning rates based on current iter and epoch,
        then update the optimizer.
        """
        for grp in self._optimizer.param_groups:
            if grp['name'] in self._learning_groups:
                param_group = self._learning_groups[grp['name']]
                grp['lr'] = param_group.get_lr(self.epoch, self.iter)


class LRStrategy(object):
    """
    Defines a learning rate function over epochs and iterations.
    """

    def __init__(self):
        pass

    def __call__(self, epoch, itr):
        raise NotImplementedError


class TriangularStrategy(object):
    """
    Symmetric triangular learning rate over each epoch with optional epoch decay.
    """

    def __init__(self, initial_lr, low_pct, total_iter, apex_cycles, epoch_decay=0.9):
        super(TriangularStrategy).__init__()
        self.total_iter = total_iter
        self.initial_lr = initial_lr
        self.low_pct = low_pct
        self.epoch_decay = epoch_decay
        self.apex_cycles = max(1, apex_cycles)

    def __call__(self, epoch, itr):
        min_lr = self.initial_lr * self.low_pct
        max_lr = self.initial_lr
        min_lr = min_lr * (self.epoch_decay ** epoch)
        max_lr = max_lr * (self.epoch_decay ** epoch)
        lr = min_lr + (max_lr - min_lr) * \
                      TriangularStrategy._rel_val(itr, self.total_iter, self.apex_cycles)
        return lr

    @staticmethod
    def _rel_val(iteration, stepsize, apex_cycles):
        it = iteration % stepsize
        scale_fn = lambda x: 1.
        if it > apex_cycles:
            x = 1 - (stepsize - it) / (stepsize - apex_cycles)
        else:
            x = 1 - it / apex_cycles
        return max(0, (1 - x)) * scale_fn(it)