import torch

from .functions import _Function
from .utils import check_random_state


class Chrom():
    """
    Chrom是一个ADF或者主程序，单个Chrom由多个gene组成，分为头部和尾部，单个gene就是单个符号如函数、常数、特征点
    feature_number: 0~i, 可以代表x0~xi，也可以代表ADF0~ADFi，具体取决于上层individual的策略
    const_range: 常数范围
    program: 自己构造的program公式，如果为None，则会随机产生
    """
    def __init__(self,
                 function_set,
                 feature_number,
                 const_range,
                 head_length,
                 tail_length,
                 random_state=None,
                 program=None
                 ):
        if tail_length < head_length+1:
            raise ValueError('length of tail %d should be at least equal to headLength+1 %d.' % (tail_length, head_length+1))

        self.function_set = function_set
        self.feature_number = feature_number
        self.arities = [function.arity for function in function_set]
        self.const_range = const_range
        self.head_length = head_length
        self.tail_length = tail_length
        self.program = program

        if self.program is None:
            self.program = self.build_program(random_state)

    def build_program(self, random):

        random = check_random_state(random)

        h = self.head_length
        l = self.tail_length
        random_length = self.feature_number + len(self.function_set)

        program = []
        # 构造头部，包含函数、特征点、常数
        for i in range(h):
            choice = random.randint(random_length)
            # 添加函数
            if choice < len(self.function_set):
                function = self.function_set[random.randint(len(self.function_set))]
                program.append(function)
            # 添加特征点或者常数
            else:
                terminal = self._get_terminal(random, self.feature_number, self.const_range)
                program.append(terminal)
                # 首位为特征点或者常数，不必继续构造
                if i == 0:
                    [program.append(0) for _ in range(h - 1 + l)]
                    return program

        # 构造尾部，只包含特征点、常数
        for i in range(l):
            program.append(self._get_terminal(random, self.feature_number, self.const_range))

        return program

    def execute(self, X):
        """
        执行此program
        """
        node = self.program[0]
        if isinstance(node, torch.FloatTensor):
            return node
        if isinstance(node, int):
            return X[:, node:node+1]

        # 得到实际最终的program长度
        valid_length = self._get_valid_length()
        program_values = [self.program[i] for i in range(valid_length)]
        i = valid_length - 1
        while i >= 0:
            symbol = self.program[i]
            arity = self._get_symbol_arity(symbol)
            if arity > 0:
                args = program_values[valid_length-arity:valid_length]
                for j in range(len(args)):
                    if isinstance(args[j], int):
                        args[j] = X[:, args[j]:args[j]+1]
                program_values[i] = symbol(*args)
                valid_length -= arity
            i -= 1
        return program_values[0]

    def get_gene(self, pos):
        return self.program[pos]

    def set_gene(self, pos, new_gene):
        self.program[pos] = new_gene

    def set_ramdom_gene(self, pos, random_state):
        random_length = self.feature_number + len(self.function_set)
        # 如果gene位置在头部区域，显然可以设置函数、特征点或者常数
        if pos < self.head_length:
            choice = random_state.randint(random_length)
            # 随机选择函数
            if choice < len(self.function_set):
                gene = self.function_set[random_state.randint(len(self.function_set))]
            # 随机选择特征点或者常数
            else:
                gene = self._get_terminal(random_state, self.feature_number, self.const_range)
        # gene位置在尾部区域，只能设置特征点或者常数
        else:
            gene = self._get_terminal(random_state, self.feature_number, self.const_range)
        self.set_gene(pos, gene)

    # 随机获取终端点即特征点或者常数
    def _get_terminal(self, random, feature_number, const_range):
        terminal = random.randint(feature_number + 1)
        # 特征点
        if terminal < feature_number:
            terminal = random.randint(feature_number)
        # 常数
        else:
            terminal = torch.tensor(random.uniform(*const_range))
        return terminal

    def _get_valid_length(self):
        valid_length, i = 1, 0
        while i < valid_length:
            symbol = self.program[i]
            valid_length += self._get_symbol_arity(symbol)
            i += 1

        return valid_length

    def _get_symbol_arity(self, symbol):
        if isinstance(symbol, _Function):
            return symbol.arity
        return 0

    def __str__(self):
        s = '['
        for symbol in self.program:
            if not isinstance(symbol, int) and not isinstance(symbol, torch.FloatTensor):
                s += symbol.name + ', '
            else:
                s += str(symbol) + ', '
        s += ']'
        return s
