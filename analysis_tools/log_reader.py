import re
        

class SingleLogReader():
    """
    SingleLogReader 类用于解析和处理单个日志文件，提取特定格式的信息，
    并根据进程的 rank 和 epoch 将信息聚合。如果指定了 rank，
    则只处理和保存该特定 rank 的日志信息。

    Attributes:
        filepath (str): 日志文件的路径。
        target_rank (int or None): 要处理的特定 rank，如果为 None，则处理所有 ranks。
        logs (dict): 存储解析后的日志数据，按 epoch 组织。
    """

    def __init__(self, filepath, target_rank=0):
        """
        初始化 SingleLogReader 类实例。

        Args:
            filepath (str): 日志文件的路径。
            target_rank (int or None): 要处理的特定 rank，如果为 None，则处理所有 ranks。
        """
        self.filepath = filepath
        self.target_rank = target_rank
        self.logs = {}
        self.__read_log()

    def __read_log(self):
        """
        读取日志文件，将日志信息存储在 self.logs 中。
        """
        epoch = None
        rank = None
        with open(self.filepath, 'r') as f:
            for line in f:
                # 解析日志信息，match '[hh:mm:ss e number r rank] message'
                match = re.match(r'\[(\d+:\d+:\d+) e (\d+) r (\d+)\] (.*)', line)
                if match is not None:
                    # 包含时间信息的行
                    cur_time, epoch, rank, msg = match.groups()
                    rank = int(rank)
                    if rank == self.target_rank:
                        # 如果是想要的rank，加上这一行
                        if epoch not in self.logs:
                            self.logs[epoch] = {}
                        self.logs[epoch][cur_time] = msg.strip() + '\n'
                else:
                    # 不是想要的rank，跳过
                    if rank is None or rank != self.target_rank:
                        continue
                    # 否则加上这一行
                    else:
                        self.logs[epoch][cur_time] += line.strip() + '\n'
                        
    def __str__(self):
        strings = ""
        for e, t in self.logs.items():
            strings += "epoch: " + e + "\n"
            for t, m in t.items():
                strings += "\t" + t + ": " + m + "\n"
        return strings
    

# import time

class LogReader():
    """
    LogReader 类用于解析和处理日志文件，提取特定格式的信息，
    并根据进程的 rank 和 epoch 将信息聚合。

    Attributes:
        filepath (str): 日志文件的路径。
        logs (dict): 存储解析后的日志数据，按 epoch 组织。
    """

    def __init__(self, filepath):
        """
        初始化 LogReader 类实例。

        Args:
            filepath (str): 日志文件的路径。
        """
        self.filepath = filepath
        self.logs = {}
        self.__read_log()
        # self.logs = sort(self.logs.keys())

    def __read_log(self):
        """
        读取日志文件，将日志信息存储在 self.logs 中。
        """
        epoch = None
        rank = None
        with open(self.filepath, 'r') as f:
            for line in f:
                # 解析日志信息，match '[hh:mm:ss e number r rank] message'
                match = re.match(r'\[(\d+:\d+:\d+) e (\d+) r (\d+)\] (.*)', line)
                if match is not None:
                    # 包含时间信息的行
                    cur_time, epoch, rank, msg = match.groups()
                    rank = int(rank)
                    if rank not in self.logs:
                        self.logs[rank] = {}
                    if epoch not in self.logs[rank]:
                        self.logs[rank][epoch] = {}
                    self.logs[rank][epoch][cur_time] = msg.strip() + '\n'
                else:
                    self.logs[rank][epoch][cur_time] += line.strip() + '\n'
                    
    def __str__(self):
        """
        将日志信息转换为字符串。
        """
        strings = ""
        for r, e in self.logs.items():
            strings += "rank: " + str(r) + "\n"
            for e, t in e.items():
                strings += "\tepoch: " + e + "\n"
                for t, m in t.items():
                    strings += "\t\t" + t + ": " + m + "\n"
        return strings