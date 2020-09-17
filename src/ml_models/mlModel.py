from abc import ABC, abstractmethod
from src.ml_models_engine.DataRepo import DataRepo
from src.enums.model_metrics import ModelMetrics
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

class MLModel(ABC):

    def __init__(self, *args, **kwargs):
        self.smell = None
        self.skModel = None
        self.dataRepo = None
        self.useGridSearch = True
        self.useRandomSearch = False

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

    def debugPrintMetrics(self, resultingModel):
        # Test accuracy

        print("\nThe test accuracy is: ")
        print("training data: " + str(accuracy_score(self.dataRepo.trainingData['y'], resultingModel.predict(self.dataRepo.trainingData['x']))))
        print("testing data: " + str(accuracy_score(self.dataRepo.testingData['y'], resultingModel.predict(self.dataRepo.testingData['x']))))

        # Test f1 score

        print("\nThe test f1 score is: ")
        print("training data: " + str(f1_score(self.dataRepo.trainingData['y'], resultingModel.predict(self.dataRepo.trainingData['x']), average='micro')))
        print("testing data: " + str(f1_score(self.dataRepo.testingData['y'], resultingModel.predict(self.dataRepo.testingData['x']), average='micro')))
