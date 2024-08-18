import numpy as np

class GreedySearchDecoder(object):

    def __init__(self, symbol_set):
        """
        初始化实例变量

        参数:
        -----------

        symbol_set [list[str]]:
            所有的符号（不包括空白符号的词汇表）

        """

        self.symbol_set = symbol_set


    def decode(self, y_probs):
        """
        执行贪婪搜索解码

        输入:
        -----

        y_probs [np.array, 维度=(len(symbols) + 1, seq_length, batch_size)]
            符号的概率分布，注意对于第一个部分的批次大小始终为1，
            但如果你计划将实现用于第二部分，则需要考虑批次大小

        返回:
        -------

        decoded_path [str]:
            压缩后的符号序列，即去除空白符号或重复符号后的序列

        path_prob [float]:
            贪婪路径的前向概率

        """

        decoded_path = []  # 用于存储解码后的路径
        blank = 0  # 空白符号的索引
        path_prob = 1  # 初始化路径概率为1

        # 1. 遍历序列长度 - len(y_probs[0])
        # 2. 遍历符号概率
        # 3. 通过与当前最大概率相乘来更新路径概率
        # 4. 选择最可能的符号并附加到解码路径
        # 5. 压缩序列（在循环内或循环外完成）

        symbols_len, seq_len, batch_size = y_probs.shape  # 获取符号长度、序列长度和批次大小
        self.symbol_set = ["-"] + self.symbol_set  # 将空白符号添加到符号集合的开头
        for batch_itr in range(batch_size):
            
            path = " "  # 初始化路径为空格
            path_prob = 1  # 初始化路径概率为1
            for i in range(seq_len):
                max_idx = np.argmax(y_probs[:, i, batch_itr])  # 找到当前时间步概率最大的符号的索引
                path_prob *= y_probs[max_idx, i, batch_itr]  # 更新路径概率
                if path[-1] != self.symbol_set[max_idx]:  # 如果当前符号与前一个符号不同
                    path += self.symbol_set[max_idx]  # 将符号添加到路径中
        
            path = path.replace('-', '')  # 移除路径中的空白符号
            decoded_path.append(path[1:])  # 将路径添加到解码后的路径列表中

        return path[1:], path_prob  # 返回解码后的路径和路径概率


class BeamSearchDecoder(object):

    def __init__(self, symbol_set, beam_width):
        """
        初始化实例变量

        参数:
        -----------

        symbol_set [list[str]]:
            所有的符号（不包括空白符号的词汇表）

        beam_width [int]:
            光束宽度，用于选择扩展的前k个假设

        """

        self.symbol_set = symbol_set
        self.beam_width = beam_width

    def decode(self, y_probs):
        """
        执行光束搜索解码

        输入:
        -----

        y_probs [np.array, 维度=(len(symbols) + 1, seq_length, batch_size)]
			符号的概率分布，注意对于第一个部分的批次大小始终为1，
            但如果你计划将实现用于第二部分，则需要考虑批次大小

        返回:
        -------
        
        forward_path [str]:
            拥有最佳路径分数（前向概率）的符号序列

        merged_path_scores [dict]:
            所有最终合并路径及其分数

        """
        self.symbol_set = ['-'] + self.symbol_set  # 将空白符号添加到符号集合的开头
        symbols_len, seq_len, batch_size = y_probs.shape  # 获取符号长度、序列长度和批次大小
        bestPaths = dict()  # 存储当前最佳路径
        tempBestPaths = dict()  # 存储临时的最佳路径
        bestPaths['-'] = 1  # 初始化最佳路径为空白符号，分数为1

        # 遍历序列长度
        for t in range(seq_len):
            sym_probs = y_probs[:, t]  # 获取当前时间步的符号概率分布
            tempBestPaths = dict()  # 重置临时路径

            # 遍历当前的最佳路径
            for path, score in bestPaths.items():

                # 遍历所有符号
                for r, prob in enumerate(sym_probs):
                    new_path = path  # 初始化新路径为当前路径

                    # 更新新路径
                    if path[-1] == '-':  # 如果当前路径的最后一个符号是空白符号
                        new_path = new_path[:-1] + self.symbol_set[r]  # 替换最后一个符号为当前符号
                    elif (path[-1] != self.symbol_set[r]) and not (t == seq_len-1 and self.symbol_set[r] == '-'):
                        new_path += self.symbol_set[r]  # 如果当前符号与最后一个符号不同，且不是空白符号，则附加当前符号

                    # 在临时路径中更新概率
                    if new_path in tempBestPaths:
                        tempBestPaths[new_path] += prob * score  # 累加路径概率
                    else:
                        tempBestPaths[new_path] = prob * score  # 初始化路径概率
                    

            # 获取前k个最佳路径并重置最佳路径
            if len(tempBestPaths) >= self.beam_width:
                bestPaths = dict(sorted(tempBestPaths.items(), key=lambda x: x[1], reverse=True)[:self.beam_width])

        # 获取得分最高的路径
        bestPath = max(bestPaths, key=bestPaths.get)
        finalPaths = dict()
        for path, score in tempBestPaths.items():
            if path[-1] == '-':
                finalPaths[path[:-1]] = score  # 移除路径末尾的空白符号
            else:
                finalPaths[path] = score  # 保持路径原样
        return bestPath, finalPaths  # 返回最佳路径和所有最终合并的路径及其分数
