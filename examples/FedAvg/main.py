import sys
sys.path.append(".") # Adds higher directory to python modules path.
sys.path.append("..") # Adds higher directory to python modules path.

from FLamingo.core.runner import Runner
# from FLamingo.datasets import generate_cifar10


if __name__ == "__main__":
    runner = Runner(cfg_file='./config.yaml')
    runner.run()