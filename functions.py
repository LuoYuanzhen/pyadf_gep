import six

import torch

__all__ = ['make_function']


class _Function(object):
    def __init__(self, function, name, arity):
        self.function = function
        self.name = name
        self.arity = arity

    def __call__(self, *args):
        return self.function(*args)


def make_function(function, name, arity):
    if not isinstance(arity, int):
        raise ValueError('arity must be an int, got %s' % type(arity))
    if not isinstance(name, six.string_types):
        raise ValueError('name must be a string, got %s' % type(name))

    # Check output shape
    args = [torch.ones(10) for _ in range(arity)]
    try:
        function(*args)
    except ValueError:
        raise ValueError('supplied function %s does not support arity of %d.'
                         % (name, arity))
    if not hasattr(function(*args), 'shape'):
        raise ValueError('supplied function %s does not return a tensor array.'
                         % name)
    if function(*args).shape != (10,):
        raise ValueError('supplied function %s does not return same shape as '
                         'input vectors.' % name)

    # Check closure for zero & negative input arguments
    args = [torch.zeros(10) for _ in range(arity)]
    if not torch.all(torch.isfinite(function(*args))):
        raise ValueError('supplied function %s does not have closure against '
                         'zeros in argument vectors.' % name)
    args = [-1 * torch.ones(10) for _ in range(arity)]
    if not torch.all(torch.isfinite(function(*args))):
        raise ValueError('supplied function %s does not have closure against '
                         'negatives in argument vectors.' % name)

    return _Function(function, name, arity)


def _protected_division(x1, x2):
    """Closure of division (x1/x2) for zero denominator."""
    one = torch.tensor(1.0)
    return torch.where(torch.abs(x2)>0.001, torch.divide(x1, x2), one)


def _protected_sqrt(x1):
    """Closure of square root for negative arguments."""
    return torch.sqrt(torch.abs(x1))


def _protected_log(x1):
    """Closure of log for zero arguments."""
    zero = torch.tensor(0.0)
    return torch.where(torch.abs(x1)>0.001, torch.log(torch.abs(x1)), zero)


def _protected_inverse(x1):
    """Closure of log for zero arguments."""
    zero = torch.tensor(0.0)
    return torch.where(torch.abs(x1)>0.001, 1. / x1, zero)


def _square(x):
    return torch.pow(x, 2)


def _cube(x):
    return torch.pow(x, 3)


add2 = make_function(function=torch.add, name='add', arity=2)
sub2 = make_function(function=torch.subtract, name='sub', arity=2)
mul2 = make_function(function=torch.multiply, name='mul', arity=2)
div2 = make_function(function=_protected_division, name='div', arity=2)
sqrt1 = make_function(function=_protected_sqrt, name='sqrt', arity=1)
log1 = make_function(function=_protected_log, name='log', arity=1)
neg1 = make_function(function=torch.negative, name='neg', arity=1)
inv1 = make_function(function=_protected_inverse, name='inv', arity=1)
abs1 = make_function(function=torch.abs, name='abs', arity=1)
max2 = make_function(function=torch.maximum, name='max', arity=2)
min2 = make_function(function=torch.minimum, name='min', arity=2)
sin1 = make_function(function=torch.sin, name='sin', arity=1)
cos1 = make_function(function=torch.cos, name='cos', arity=1)
tan1 = make_function(function=torch.tan, name='tan', arity=1)
square1 = make_function(function=_square, name='square', arity=1)
cube1 = make_function(function=_cube, name='cube', arity=1)

_function_map = {'add': add2,
                 'sub': sub2,
                 'mul': mul2,
                 'div': div2,
                 'sqrt': sqrt1,
                 'log': log1,
                 'abs': abs1,
                 'neg': neg1,
                 'inv': inv1,
                 'max': max2,
                 'min': min2,
                 'sin': sin1,
                 'cos': cos1,
                 'tan': tan1,
                 'square': square1,
                 'cube': cube1}
