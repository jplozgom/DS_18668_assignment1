from enum import Enum, unique

@unique
class ModelMetrics(Enum):

    """List of available metrics to evaluate a models in the system"""

    ACURRACY = "acurracy"
    F1 = "f1_micro"
    PRECISION = "precision"
    RECALL = "recall"

    def label(self):
        """ The label used for each metric in a list """
        return str(self.value)
