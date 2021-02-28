import torch
import six
import copy


from .individual import Individual
from .utils import check_random_state
from .functions import _function_map
from .functions import _Function


"""
    adf_labels:自定义的给每个ADF做损失的标签值，若不指定，则每个个体的适应度为正常的ADF-GEP计算方式，默认None
    nested:bool型，是否嵌套，若嵌套，则后一个ADF/main的特征值为前一个ADF的输出，默认False
    
"""
class Population():
    def __init__(self,
                 function_set,
                 adf_num,
                 feature_number,
                 const_range,
                 population_size,
                 generation_size,
                 adf_headlength,
                 adf_taillength,
                 main_headlength,
                 main_taillength,
                 mutation_rate=0.4,
                 onePointRecom_rate=0.3,
                 twoPointRecom_rate=0.2,
                 bestIndiv_number=10,
                 random_state=None,
                 adf_labels=None,
                 nested=None,
                 population=None):
        self.adf_num = adf_num
        self.feature_number = feature_number
        self.const_range = const_range
        self.population_size = population_size
        self.generation_size = generation_size
        self.adf_headlength = adf_headlength
        self.adf_taillength = adf_taillength
        self.main_headlength = main_headlength
        self.main_taillength = main_taillength
        self.mutation_rate = mutation_rate
        self.onePointRecom_rate = onePointRecom_rate
        self.twoPointRecom_rate = twoPointRecom_rate
        self.bestIndiv_number = bestIndiv_number
        self.random_state = random_state
        self.adf_labels = adf_labels
        self.nested = nested
        self.population = population

        self.best_individuals = []
        self.function_set = []
        for function in function_set:
            if isinstance(function, six.string_types):
                if function not in _function_map:
                    raise ValueError('invalid function name %s found in `function_set`.' % (function))
                self.function_set.append(_function_map[function])
            elif isinstance(function, _Function):
                self.function_set.append(function)
            else:
                raise ValueError('invalid type %s found in `function_set`.' % (type(function)))

    def initialization(self):
        population = []
        for i in range(self.population_size):
            indiv = Individual(function_set=self.function_set,
                               const_range=self.const_range,
                               feature_number=self.feature_number,
                               adf_num=self.adf_num,
                               adf_headlength=self.adf_headlength,
                               adf_taillength=self.adf_taillength,
                               main_headlength=self.main_headlength,
                               main_taillength=self.main_taillength,
                               adf_labels=self.adf_labels,
                               nested=self.nested
                               )
            population.append(indiv)
        return population

    def start_evolute(self, X, y):
        is_best = False
        if self.population is None:
            self.population = self.initialization()
        # 计算适应度
        self.evaluate_fitness(X, y)
        # 适应度从小到大排序，适应度越小越好
        self.sort_individuals()
        # 保存最优个体
        self.update_bestIndividuals()

        for gen in range(self.generation_size):
            print(self.get_report(gen+1))

            if self.best_individuals[0].fitness <= 0.01:
                is_best = True
                break

            # 变异
            self.one_point_reorganization()
            self.two_point_reorganization()
            self.mutation()

            # 计算适应度以及保留最优个体
            self.evaluate_fitness(X, y)
            self.sort_individuals()

            # 赌轮盘
            self.select()

            # 排序以及更新最优个体
            self.sort_individuals()
            self.update_bestIndividuals()

        print(self.get_evolute_result(is_best))

    def evaluate_fitness(self, X, y):
        for indiv in self.population:
            fitness = indiv.get_fitness(X, y)
            if isinstance(fitness, list):
                indiv.fitness = sum(fitness) / len(fitness)
                indiv.loss_list = fitness
            else:
                indiv.fitness = fitness
                indiv.loss_list = []
                indiv.loss_list.append(fitness)

    def sort_individuals(self):
        def _take_fitness(element):
            return element.fitness

        self.population.sort(key=_take_fitness)

    def update_bestIndividuals(self):
        if len(self.best_individuals) == 0:
            for i in range(self.bestIndiv_number):
                indiv = self.population[i]
                self.best_individuals.append(copy.deepcopy(indiv))
        else:
            j = 0
            for i in range(self.bestIndiv_number):
                indiv = self.best_individuals[i]
                if indiv.fitness > self.population[j].fitness:
                    self.best_individuals[i] = copy.deepcopy(self.population[j])
                    j += 1

    def one_point_reorganization(self):
        self.random_state = check_random_state(self.random_state)

        # 这里不用self.population_size的原因是此成员变量仅仅代表最初的种群数量
        # 每一代可能因为轮盘策略而损失掉一部分个体，所以个体数量时刻在变化
        population_size = len(self.population)
        for i in range(population_size):
            rate = self.random_state.uniform(0, 1)
            if rate < self.onePointRecom_rate:
                father = self.random_state.randint(population_size)
                mother = self.random_state.randint(population_size)

                father, mother = self.population[father], self.population[mother]
                gene_pos = self.random_state.randint(father.indiv_length)
                for p in range(gene_pos):
                    gene = father.get_gene(p)
                    father.set_gene(p, mother.get_gene(p))
                    mother.set_gene(p, gene)

    def two_point_reorganization(self):
        self.random_state = check_random_state(self.random_state)

        population_size = len(self.population)
        for i in range(population_size):
            rate = self.random_state.uniform(0, 1)
            if rate < self.twoPointRecom_rate:
                father = self.random_state.randint(population_size)
                mother = self.random_state.randint(population_size)

                father, mother = self.population[father], self.population[mother]
                gene_start = self.random_state.randint(father.indiv_length)
                gene_end = self.random_state.randint(father.indiv_length)
                for p in range(gene_start, gene_end):
                    gene = father.get_gene(p)
                    father.set_gene(p, mother.get_gene(p))
                    mother.set_gene(p, gene)

    def mutation(self):
        self.random_state = check_random_state(self.random_state)

        population_size = len(self.population)
        for i in range(population_size):
            individual = self.population[i]
            indiv_length = individual.indiv_length
            for j in range(indiv_length):
                rate = self.random_state.uniform(0, 1)
                if rate < self.mutation_rate:
                    individual.set_random_gene(j, self.random_state)

    # 赌轮盘策略选择
    def select(self):
        # 找到最小的损失
        min_fitness = self.population[0].fitness

        population_size = len(self.population)
        for i in range(1, population_size):
            if self.population[i].fitness < min_fitness:
                min_fitness = self.population[i].fitness

        # 计算所有的轮盘值
        for i in range(population_size):
            self.population[i].wheel_value = min_fitness / self.population[i].fitness

        # 构造新的种群
        new_population = []
        new_population.extend(self.best_individuals)

        total_wheelvalue = sum([indiv.wheel_value for indiv in self.population])

        # 每个个体的概率
        if total_wheelvalue == 0:
            rate = 1 / population_size
            rates = [rate for _ in range(population_size)]
        else:
            rates = [indiv.wheel_value / total_wheelvalue for indiv in self.population]

        # 赌轮盘
        wheel = [rates[0]]
        for i in range(1, population_size):
            wheel.append(wheel[i-1] + rates[i])

        # 选择
        for i in range(self.bestIndiv_number, population_size):
            rate = self.random_state.uniform(0, 1)
            for j in range(population_size):
                if rate < wheel[j]:
                    break
            new_population.append(self.population[j])

        self.population = new_population

    def get_report(self, gens):
        def _get_indivReport(individual, num):
            s = 'best individual %d:\n%s \n' % (num, individual.__str__())
            s = s + 'fitness : %f, loss list : %s' % (individual.fitness, individual.loss_list)
            return s

        report = 'generation %d \n' % (gens)
        report = report + _get_indivReport(self.best_individuals[0], 0) + '\n'
        report = report + _get_indivReport(self.best_individuals[3], 3) + '\n'
        report = report + _get_indivReport(self.best_individuals[6], 6) + '\n'
        report = report + 'now population size is:' + str(len(self.population)) + '\n'

        return report

    def get_evolute_result(self, success):
        title = 'Reach the best, best individual fitness: ' if success else 'Not optimal, best individual fitness: '
        result = title + str(self.best_individuals[0].fitness) + '\nbest individual: \n%s' % (self.best_individuals[0].__str__())
        if self.adf_labels is not None:
            result = result + 'best individual loss list: %s' % (self.best_individuals[0].loss_list)
        return result