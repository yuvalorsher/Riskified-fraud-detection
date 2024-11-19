from __future__ import annotations
import numpy as np
import pandas as pd
from task_infra.experiment_pipeline import Experiment
from task_infra.evaluations import Evaluator


class Reporter:
    def __init__(self):
        self.contents = []

    def add_paragraph(self, text: str):
        """Adds a paragraph of text."""
        self.contents.append(f"<p>{text}</p>")

    def add_header(self, text: str, level: int = 1):
        """Adds a header of specified level (1 to 3)."""
        if level not in [1, 2, 3]:
            raise ValueError("Header level must be 1, 2, or 3.")
        self.contents.append(f"<h{level}>{text}</h{level}>")

    def add_table(self, table: pd.DataFrame | np.ndarray):
        """Adds a table from a DataFrame or a NumPy array."""
        if isinstance(table, pd.DataFrame):
            html_table = table.to_html(index=False, escape=False, border=1)
        elif isinstance(table, np.ndarray):
            html_table = pd.DataFrame(table).to_html(index=False, escape=False, border=1)
        else:
            raise TypeError("Input must be a pandas DataFrame or a numpy array.")
        self.contents.append(html_table)

    def add_raw_html(self, html: str):
        """Adds a raw html text without adding brackets"""
        self.contents.append(html)

    def report(self, filename: str):
        """Saves the contents to an HTML file."""
        html_content = "<html><body>\n" + "\n".join(self.contents) + "\n</body></html>"
        with open(filename, "w") as file:
            file.write(html_content)
        print(f"Report saved to {filename}")

    @staticmethod
    def create_report_from_experiment(experiment: Experiment) -> Reporter:
        def _classification_dict_to_html(classification_metrics: dict) -> str:
            metrics_df = (pd.DataFrame(classification_metrics)
                          .T
                          .drop('confusion_matrix')  # IF in metrics, has to be speicially treated
                          )
            return metrics_df.style.background_gradient(axis=1).format('{:.3f}').to_html()

        evaluation_step: Evaluator = experiment.get_subtask('Evaluation')
        classification_metrics = evaluation_step.outputs[evaluation_step.classification_metrics_key]
        required_fee = evaluation_step.outputs[evaluation_step.required_fee_key]
        required_ratio = evaluation_step.params['cost_of_cb_to_revenue_ratio']

        reporter = Reporter()
        reporter.add_header('Model Performance Report', level=1)
        reporter.add_header('Classification Metrics', level=2)
        reporter.add_raw_html(_classification_dict_to_html(classification_metrics))
        reporter.add_header("Required Fee")
        reporter.add_paragraph(
            f"Requested ratio of cost of CB to revenue: {required_ratio:.2}"
        )
        reporter.add_paragraph(f"To get this ratio, required fee must be: {required_fee:.1%} of transactions.")
        return reporter
