import sys
sys.path.append('../FLamingo/')

from FLamingo.runner import Runner

if __name__ == "__main__":
    # init Runner
    runner = Runner(cfg_file='./config.yaml')
    runner.run()