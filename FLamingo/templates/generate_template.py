# Generate template project dir to quickly hands on
import os
from argparse import ArgumentParser
import shutil


def generate_project_dir(project_dir):
    """
    Create the project directory
    """
    os.makedirs(project_dir, exist_ok=True)

    # Create the main.py file
    with open(os.path.join(project_name, 'main.py'), 'w') as file:
        file.write('# This is the main entry point of your project')

    # Create the README.md file
    with open(os.path.join(project_name, 'README.md'), 'w') as file:


if __name__ == '__main__':
