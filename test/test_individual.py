import torch

import copy

from adf_gep.functions import _function_map
from adf_gep.individual import Individual


def show_individual(individual):
    for chrom in individual:
        print(chrom)


X = torch.tensor([[0, 1, 0], [3.14, 2, 3.14], [1.57, 3, 1.57]])
y = torch.sin(X[:, 0:1]) + torch.pow(X[:, 1:2], 2) + torch.cos(X[:, 2:3])

function_set = [_function_map['add'], _function_map['mul'], _function_map['sin'],
                _function_map['cos'], _function_map['div'], _function_map['log']]
adf_num = 3
adf_headlength = 5
main_headlength = 6
individual = Individual(function_set=function_set,
                        const_range=(-2, 2),
                        feature_number=3,
                        adf_num=adf_num,
                        adf_headlength=adf_headlength,
                        adf_taillength=adf_headlength+1,
                        main_headlength=main_headlength,
                        main_taillength=main_headlength+1,
                        nested=True
                        )

chrom = individual.individual[0]
print(chrom, individual.individual[0])

chrom.set_gene(0, 1)
print(chrom, individual.individual[0])

loss = individual.get_fitness(X, y)
print(loss)


