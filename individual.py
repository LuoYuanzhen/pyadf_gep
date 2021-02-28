import torch

from .chromsome import Chrom

from .utils import check_random_state
from .utils import MSELoss

"""
    Individual个体，由多个Chrom染色体组成，每个Chrom染色体表示一个ADF或者main program
    feature_number:数据集X的特征数i, x0~xi
    adf_num:ADF的个数，不包含主程序
    adf_labels:对每个ADF的值做损失差的标签list,长度必须与adf_num一致
               若为None,则Individual的适应度为最终主程序的结果与数据集标签作损失
               若指定，则Individual的最终适应度为所有ADF与adf_labels的损失、主程序与数据集标签的损失之和
    individual:类型list，包含ADFs与主程序
    
    get_gene(pos): 第pos的基因，即单个符号，例如函数、常数、特征点
"""


class Individual():
    def __init__(self,
                 function_set,
                 const_range,
                 feature_number,
                 adf_num,
                 adf_headlength,
                 adf_taillength,
                 main_headlength,
                 main_taillength,
                 random_state=None,
                 adf_labels=None,
                 nested=False,
                 individual=None):
        self.function_set = function_set
        self.feature_number = feature_number
        self.const_range = const_range
        self.adf_num = adf_num
        self.adf_headlength = adf_headlength
        self.adf_taillength = adf_taillength
        self.main_headlength = main_headlength
        self.main_taillength = main_taillength
        self.adf_labels = adf_labels
        self.nested = nested
        self.individual = individual

        self.indiv_length = self.adf_num * (self.adf_headlength + self.adf_taillength) + self.main_headlength + self.main_taillength
        self.fitness = None
        self.wheel_value = None
        self.loss_list = []

        if self.adf_labels is not None and len(adf_labels) != adf_num:
            raise ValueError('length of chrom_labels %d should be equal to adf_num %d' % (len(adf_labels), adf_num))

        if self.individual is None:
            self.individual = self.build_individual(random_state)

    def build_individual(self, random_state):
        random_state = check_random_state(random_state)

        individual = []
        # 构造ADFs
        feature_number = self.feature_number
        for i in range(self.adf_num):
            # 对于是否嵌套，构造ADF的feature_number不一样
            # 对于嵌套而言：第一个ADF的feature_number为X的特征点数，往后每一个ADF的feature_number为1
            if self.nested and i > 0:
                feature_number = 1
            individual.append(Chrom(function_set=self.function_set,
                                      feature_number=feature_number,
                                      const_range=self.const_range,
                                      head_length=self.adf_headlength,
                                      tail_length=self.adf_taillength))

        # 构造main program
        feature_number = 1 if self.nested else self.adf_num
        individual.append(Chrom(function_set=self.function_set,
                                  feature_number=feature_number,
                                  const_range=self.const_range,
                                  head_length=self.main_headlength,
                                  tail_length=self.main_taillength))

        return individual

    def get_fitness(self, X, y):
        """
        X为ADF0的输入，y为最终主程序的标签
        如果指定了adf_labels，那么返回类型为list的适应度列表
        如果不指定，则返回主程序与标签的损失
        """
        value = X
        # value_matrix每一列代表每一个ADF的执行值
        value_matrix = []
        for chrom in self.individual[:-1]:
            if self.nested:
                value = chrom.execute(value)
            else:
                value = chrom.execute(X)
            if len(value.shape) == 0:
                value = torch.full((X.shape[0], 1), value)
            value_matrix.append(value)

        value_matrix = torch.cat(value_matrix, 1) # 扩展第1维（列），合并为一个矩阵

        main_chrom = self.individual[-1]
        if self.nested:
            main_value = main_chrom.execute(value)
        else:
            main_value = main_chrom.execute(value_matrix)

        # 将单值填充至与y一样的维度，避免执行MSELoss时广播机制的警告信息
        if len(main_value.shape) == 0:
            main_value = torch.full(y.shape, main_value)

        main_loss = MSELoss(main_value, y)
        if self.adf_labels is None:
            return main_loss

        loss_list = []
        for i in range(len(self.adf_labels)):
            loss_list.append(MSELoss(self.adf_labels[i], value_matrix[:, i]))
        loss_list.append(main_loss)

        return loss_list

    def get_specific_location(self, pos):
        if pos >= self.indiv_length:
            raise ValueError('position %d bigger than the max length of individual %d' % (pos, self.indiv_length))

        adf_length = self.adf_headlength + self.adf_taillength

        chrom_count = pos // adf_length
        gene_count = pos % adf_length

        # 如果染色体的位置大于ADF的数值，说明，pos位置的gene在最后一个chrom即main program上
        if chrom_count > self.adf_num:
            gene_count += (chrom_count-self.adf_num) * adf_length
            chrom_count = -1
        return chrom_count, gene_count

    def get_gene(self, pos):
        chrom_count, gene_count = self.get_specific_location(pos)
        return self.individual[chrom_count].get_gene(gene_count)

    def set_gene(self, pos, new_gene):
        chrom_count, gene_count = self.get_specific_location(pos)
        self.individual[chrom_count].set_gene(gene_count, new_gene)

    def set_random_gene(self, pos, random_state):
        chrom_count, gene_count = self.get_specific_location(pos)
        self.individual[chrom_count].set_ramdom_gene(gene_count, random_state)

    def __str__(self):
        s = '['
        for chrom in self.individual:
            s = s + chrom.__str__() + ';\n'
        s = s + ']'
        return s
