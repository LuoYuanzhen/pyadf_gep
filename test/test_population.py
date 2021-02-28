import torch

from adf_gep.population import Population
from data_tools import io


dataPath = 'D:\\my_jupyter\\Datasets\\Dataset\\Koza\\one_v\\1.txt'
function_set = ['add', 'mul', 'square', 'cube']
adf_num, feature_number, const_range, population_size, generation_size = 3, 1, (-1, 1), 100, 1000
adf_headlength, adf_taillength, main_headlength, main_taillength = 3, 4, 4, 5

dataset = io.get_dataset(dataPath)

X = dataset[:, 0:1]
y = dataset[:, -1:]

population = Population(function_set=function_set,
                        adf_num=adf_num,
                        feature_number=feature_number,
                        const_range=const_range,
                        population_size=population_size,
                        generation_size=generation_size,
                        adf_headlength=adf_headlength,
                        adf_taillength=adf_taillength,
                        main_headlength=main_headlength,
                        main_taillength=main_taillength)
population.start_evolute(X, y)
