"""

This module provides a function to generate templates to a specified destination directory.
Usage:
from FLamingo.templates.generate_template import generate
generate(destination_dir, ['config','server','client'])

"""

from fileinput import filename
import os
import shutil
from tkinter import NO

def generate(destination_dir=None, module_name_list=None):
    """
    Generate templates to the destination directory.
    If the destination directory is not specified, the current directory will be used.
    Module name list is a list of module names to be generated. 
    If the module name list is not specified, all modules will be generated.
    
    Args:
    destination_dir: str, optional
        The destination directory to store the generated templates.
    module_name_list: list, optional
        A list of module names to be generated.
        
    Returns:
    None
    
    Raises:
    None
    """
    template_str = '_template_'
    default_name_list = ['client','config','server','network','runner']
    # 获取当前模块的文件路径
    module_path = os.path.abspath(__file__)
    module_dir = os.path.dirname(module_path)
    # 获取工作目录
    work_dir = os.getcwd()
    
    # 如果没有指定目标目录，则使用模块所在的目录
    if destination_dir is None:
        destination_dir = work_dir
    
    # 确保目标目录存在
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
    
    if module_name_list is None:
        module_name_list = default_name_list
    else:
        set_a = set(module_name_list)
        set_b = set(default_name_list)
        if set_a-set_b !=set():
            set_c = set_a-set_b
            print(f'{list(set_c)} invalid.')
            return
        
    for item in module_name_list:
        file = template_str + item
        file_name = f'{file}.txt'
        if item == 'config':
            dst_name = f'{item}.yaml'
        else:
            dst_name = f'{item}.py'
        src_file = os.path.join(module_dir, file_name)
        dst_file = os.path.join(destination_dir, dst_name)

        shutil.copy2(src_file, dst_file)
        print(f'Item "{item}" generated as "{dst_file}".')
