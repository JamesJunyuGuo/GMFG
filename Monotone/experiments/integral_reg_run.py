import pickle
import pathlib

from experiments import integral_args_parser
from experiments.integral_reg_trainer import integral_reg_run_experiment


def integral_reg_run(config):
    results, simu_config = integral_reg_run_experiment(**config)
    
    filepath = './integral_result/'+simu_config['exp_name']+"/"
    pathlib.Path(filepath).mkdir(parents=True, exist_ok=True)
    with open(filepath + 'logs.pkl', 'wb') as f:
        pickle.dump(results, f, 4)
    with open(filepath + 'config.pkl', 'wb') as f:
        pickle.dump(simu_config, f, 4)
    
    

if __name__ == '__main__':
    config = integral_args_parser.parse_config()
    integral_reg_run(config)
