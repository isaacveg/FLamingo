from uu import Error
import yaml
import os
import time
import subprocess
import platform, signal


class Runner(object):
    def __init__(self, cfg_file=None):
        assert cfg_file is not None, "No config file found."
        self.cfg = self.ParseConfig(cfg_file)
        if self.cfg is None:
            raise ValueError("No config file found.")
        config = self.cfg
        self.process = None
        
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
    
    def add_cfg(self, key, value):
        """
        Add dict to the config.
        """
        self.cfg.update({key: value})
        
    def update_cfg(self, key, value):
        """
        Update the config.
        """
        self.cfg[key] = value

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
        self.add_cfg('run_dir', run_dir)

        # Create run_dir
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)

        if config['log_path'] is None:
            log_path = os.path.join(run_dir, "logs.log")
            self.add_cfg('log_path', log_path)

        # export config.yaml to run_dir
        self.add_cfg('saved_config_dir', os.path.join(run_dir, 'config.yaml'))
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
            with open(log_path, 'w') as log:
                # self.process = subprocess.Popen(mpiexec_cmd, stdout=subprocess.STDOUT, stderr=subprocess.STDOUT)
                self.process = subprocess.Popen(mpiexec_cmd, stdout=log, stderr=subprocess.STDOUT)
                self.process.wait()
        except KeyboardInterrupt:
            print("Interrupted by user.")
            self.terminate()
        except Exception as e:
            print(e)
            self.terminate()
        finally:
            self.process = None


