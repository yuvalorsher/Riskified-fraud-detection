import json
from task_infra.experiment_pipeline import Experiment
from task_infra.reporter import Reporter

json_file = "experiment_params.json"
report_output_path = "experiment_report.html"
save_trained_model_to = "trained_model.pkl"

if __name__ == '__main__':
    with open(json_file) as f:
        params = json.load(f)

    exp = Experiment(params)
    exp.save_trained_model(save_trained_model_to)
    reporter = Reporter.create_report_from_experiment(exp)
    reporter.report(report_output_path)
