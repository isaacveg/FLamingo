from argparse import ArgumentParser
import yaml


def get_args():
    """
    Parse args and pass them to processes
    Returns:
        args: parsed arguments from yaml config file passed through cmd line
    """
    parser = ArgumentParser(description="Passing arguments to subprocesses")
    parser.add_argument('-cfg', "--config", type=str, default=None, help="use config file, will overwrite any other arguments")

    args, unknown = parser.parse_known_args()   
    if args.config is not None:
        with open(args.config, "r") as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
        for k, v in cfg.items():
            # use config to overwrite args 
            setattr(args, k, v)
    if unknown:
        print(f"Unknown args: {unknown}")
        
    SYSTEM_DEFAULTS = {
        'USE_SIM_SYSHET': False,
        'sys_het_list': None,
        'USE_TENSORBOARD': False,
        'CREATE_SUBDIR_PER_RUN': True
    }
    
    # if the SYSTEM_DEFAULTS are not in the config, set them to False
    for k, v in SYSTEM_DEFAULTS.items():
        if not hasattr(args, k):
            setattr(args, k, v)
    
    assert args.num_training_clients <= args.num_clients, f"training_clients {args.num_training_clients} cannot be more than num_clients {args.num_clients}"
    return args

