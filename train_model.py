import json
from task_infra.experiment_pipeline import Experiment

json_file = "experiment_params.json"
save_trained_model_to = "trained_model.pkl"

if __name__ == '__main__':
    with open(json_file) as f:
        params = json.load(f)

    exp = Experiment(params)
    exp.save_trained_model(save_trained_model_to)
