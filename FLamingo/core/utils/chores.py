import time
import logging
import os
from tensorboardX import SummaryWriter


def log(rank, global_round, log_str):
    """
    Log info string with time and rank. This will printed to logs.log file.
    """
    time_str = time.strftime("%H:%M:%S", time.localtime())
    print("[{} e {} r {}] ".format(time_str, global_round, rank)+
            log_str
    )
    
    
def create_logger(log_file_path):
    """
    Create logger and return it.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_file_path)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def create_recorder(event_log_dir):
    """
    Create and return a tensorboardX SummaryWriter.
    """
    if not os.path.exists(event_log_dir):
        os.makedirs(event_log_dir, exist_ok=True)
    recorder = SummaryWriter(log_dir=event_log_dir)
    return recorder
    

def merge_several_dicts(dict_list):
        """
        Merge several dicts into one
        """
        merged_dict = {}
        for dic in dict_list:
            merged_dict.update(dic)
        return merged_dict