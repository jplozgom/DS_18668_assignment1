from abc import ABC, abstractmethod
from src.ml_models_engine.DataRepo import DataRepo

class MLModel(ABC):

    def __init__(self, *args, **kwargs):
        self.smell = None
        self.skModel = None
        self.dataRepo = None

        if 'smell' in kwargs :
            self.smell = kwargs['smell']

        if 'dataRepo' in kwargs :
            self.dataRepo = kwargs['dataRepo']
        else:
            self.loadDataRepo()

    @abstractmethod
    def trainModel(self):
        pass

    def modelExists(self):
        pass

    def saveModelResults(self):
        pass

    def retrieveModelResults(self):
        pass

    def loadDataRepo(self):
        # 1. create data repo
        self.dataRepo = DataRepo(smell=self.smell)

    def loadTrainingAndTestingData(self):
        self.dataRepo.loadDataset();