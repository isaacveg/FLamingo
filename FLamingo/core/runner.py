from uu import Error
import yaml
import os
import time
import subprocess
import platform
import signal
import shutil
import sys


class Runner(object):
    def __init__(self, cfg_file=None):
        assert cfg_file is not None, "No config file found."
        self.cfg = self.ParseConfig(cfg_file)
        if self.cfg is None:
            raise ValueError("No config file found.")
        self.process = None
        self.last_run_success = False
        self.last_run_dir = None
        
    def ParseConfig(self, cfg_file):
        """
        Parse the config file and set the corresponding attributes.
        """
        # read in yaml 
        if cfg_file is None:
            return None
        with open(cfg_file, 'r') as stream:
            try:
                config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        return config
    
    def update_cfg(self, key_or_dict, value=None):
        """
        Update the config.
        
        Args:
            key_or_dict (str or dict): If a string is provided, it is used as the key 
                                    and `value` is used as the value to update the config.
                                    If a dict is provided, its key-value pairs are used 
                                    to update the config.
            value (optional): The value to update in the config if `key_or_dict` is a string.
        """
        if isinstance(key_or_dict, dict):
            # Update the config with all key-value pairs from the provided dictionary
            self.cfg.update(key_or_dict)
        else:
            # Update or add the single key-value pair to the config
            self.cfg[key_or_dict] = value

    def export_config(self, file_path):
        """
        Export the config and added settings.
        """
        with open(file_path, 'w') as f:
            yaml.dump(self.cfg, f)

    def terminate(self):
        """
        Terminate the process.
        """
        if self.process is not None:
            if platform.system() == 'Windows':
                self.process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                self.process.terminate()
                self.process.wait()
            self.process = None

    def rename_last_run_dir(self, new_name):
        """
        Rename the last run directory.
        """
        if self.last_run_dir is not None:
            if self.last_run_success:
                os.rename(self.last_run_dir, new_name)
            else:
                print("Last run was not successful, not renaming directory.")
        else:
            print("No last run directory found.")
    
    def run(self):
        """
        Use given config to run server and client.
        """
        config = self.cfg

        # set configs
        if config['CREATE_SUBDIR_PER_RUN']:
            timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            run_dir = os.path.join(config['work_dir'], timestamp)
        else:
            run_dir = config['work_dir']
        self.update_cfg('run_dir', run_dir)
        self.last_run_dir = run_dir 
        # Create run_dir
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)

        # if config['log_path'] is None:
        log_path = os.path.join(run_dir, "logs.log")
        self.update_cfg('log_path', log_path)

        # export config.yaml to run_dir
        self.update_cfg('saved_config_dir', os.path.join(run_dir, 'config.yaml'))
        self.export_config(os.path.join(run_dir, 'config.yaml'))

        # Define the command to execute with mpiexec. Use -u for immediate output
        mpiexec_cmd = [
            'mpiexec', '-n', '1', 
            'python', '-u', config['server_file'], '--config', config['saved_config_dir'], 
            ':', '-n', str(config['num_clients']), 
            'python', '-u',config['client_file'], '--config', config['saved_config_dir']
            # ,'> ' + log_path
        ]

        # Execute the command with subprocess, catch errors, and terminate the mpi processes
        try:
            # with open(log_path, 'w') as log:
            #     # self.process = subprocess.Popen(mpiexec_cmd, stdout=subprocess.STDOUT, stderr=subprocess.STDOUT)
            #     self.process = subprocess.Popen(mpiexec_cmd, stdout=log, stderr=subprocess.STDOUT)
            #     self.process.wait()
            self.process = subprocess.Popen(mpiexec_cmd, stdout=None, stderr=None)
            print("Running...")
            self.process.wait()
        except KeyboardInterrupt:
            print("Interrupted by user.")
            self.terminate()
            self.last_run_success = False
            # Ask whether remove the output directory if user input y, other input will keep the output directory
            if input("Do you want to remove the output directory? (y/n): ") == 'y':
                shutil.rmtree(run_dir)
            else:
                print(f"Output directory is saved at {run_dir}")
        except Exception as e:
            print(e)
            self.terminate()
            self.last_run_success = False
        else:
            # running success
            self.last_run_success = True
        finally:
            self.process = None
            self.last_run_dir = run_dir


