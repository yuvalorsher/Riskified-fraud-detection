from abc import ABC, abstractmethod


class Task(ABC):
    """
    A Task object that deals with part of, or the whole, training
    """
    mlflow_client: None
    outputs = dict()
    params = dict()

    @abstractmethod
    def __init__(self, params: dict):
        raise NotImplementedError()

    @abstractmethod
    def run(self):
        raise NotImplementedError()

    @staticmethod
    def get_param_hash(s):
        pass

    def check_if_cached(self):
        # param_hash = get_param_hash(self.params)
        return False

    def load_if_cached(self):
        pass

    def run_or_load_from_hash(self):
        if self.check_if_cached() is True:
            print("Experiment Found! Loading cached results.")
            self.load_if_cached()
        else:
            self.run()
