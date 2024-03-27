import time



def log(rank, global_round, log_str):
    """
    Log info string with time and rank
    """
    time_str = time.strftime("%H:%M:%S", time.localtime())
    print("[{} e {} r {}] ".format(time_str, global_round, rank)+
            log_str
    )


def merge_several_dicts(dict_list):
        """
        Merge several dicts into one
        """
        merged_dict = {}
        for dic in dict_list:
            merged_dict.update(dic)
        return merged_dict