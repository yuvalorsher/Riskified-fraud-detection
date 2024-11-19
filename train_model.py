import json
from task_infra.experiment import Experiment
from task_infra.reporter import Reporter
import pickle

json_file = "experiment_params.json"
report_savepath = "experiment_report.html"
trained_model_savepath = "trained_model.pkl"

load_experiment = False
load_experiment_path = "yuval_full_experiment.pkl"  # Will only be used if load_experiment is True
full_experiment_savepath = "full_experiment.pkl"

if __name__ == '__main__':
    if load_experiment:
        with open(load_experiment_path, 'rb') as handle:
            exp = pickle.load(handle)
    else:
        with open(json_file) as f:
            params = json.load(f)
        exp = Experiment(params)
        exp.save_experiment(full_experiment_savepath)
        exp.save_trained_model(trained_model_savepath)
    reporter = Reporter.create_report_from_experiment(exp)
    reporter.report(report_savepath)
