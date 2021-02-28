from adf_gep.chromsome import Chrom
from adf_gep.functions import _function_map

import torch

import numpy as np


function_set = [_function_map['add'], _function_map['mul'], _function_map['sin'],
                _function_map['cos'], _function_map['div'], _function_map['log']]
program = Chrom(function_set, 2, (-1, 1), 5, 6)
print('program:\n', program)

X = torch.tensor([[1, 2], [0, 1]], dtype=torch.float)
print('X:\n', X)
print('execute:\n', program.execute(X))