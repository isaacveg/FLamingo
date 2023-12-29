

class LogReader():
    def __init__(self, rank, log_file) -> None:
        with open(log_file, "r") as f:
            lines = f.readlines()
            cur_log = []
            for line in lines:
                if f'r{str(rank)}]' in line:
                    cur_log.append(line)
            self.log = cur_log
            self.log_file = log_file
    
    def __str__(self) -> str:
        logs = ""
        for line in self.log:
            logs += line
            # logs += '\n'
        return logs



if __name__ == '__main__':
    server_log = LogReader(0, '../runs/fedavg_mnist/logs.log')
    print(server_log)