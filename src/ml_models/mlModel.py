from abc import ABC, abstractmethod
import os
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
# import joblib
from src.ml_models_engine.DataRepo import DataRepo


class MLModel(ABC):

    def __init__(self, *args, **kwargs):
        self.model = None
        self.smell = None
        self.skModel = None
        self.dataRepo = None
        self.useGridSearch = True
        self.useRandomSearch = False
        self.persistModel = True

        if 'smell' in kwargs :
            self.smell = kwargs['smell']

        if 'model' in kwargs :
            self.model = kwargs['model']

        if 'dataRepo' in kwargs :
            self.dataRepo = kwargs['dataRepo']
        else:
            self.loadDataRepo()

    @abstractmethod
    def trainModel(self):
        pass

    def modelExists(self):
        pass

    def saveModel(self):
        """ PERSISTS A MODEL IN DISK """
        if self.skModel is None:
            raise SystemError('INVALID MODEL, PLEASE TRAIN')

        joblib.dump(self.skModel, self.getPklFilePath())

    def retrieveModelResults(self):
        """ RETRIEVES A MODEL FROM DISK IF THE FILE EXISTS """

        if os.path.exists(self.getPklFilePath()):
            self.skModel = joblib.load(self.getPklFilePath())
        else:
            raise FileNotFoundError("We could not find a trained model for '"+self.model.label()+"'. Please train it first")

    def getPklFilePath(self):
        cwd = os.getcwd()
        smellName = self.smell.label()
        modelName = self.model.label()
        fileName = str(modelName + "___"+smellName).lower().replace(" ", "_") + ".pkl"
        return os.path.join(cwd,"generated_models", fileName);

    def loadDataRepo(self):
        # 1. create data repo
        self.dataRepo = DataRepo(smell=self.smell)

    def loadTrainingAndTestingData(self):
        self.dataRepo.loadDataset();



    def getF1Score(self, *args, **kwargs):

        if self.skModel is None:
            raise SystemError('INVALID MODEL, PLEASE TRAIN')
        if 'data' in kwargs:
            if kwargs['data'] == "training":
                return f1_score(self.dataRepo.trainingData['y'], self.skModel.predict(self.dataRepo.trainingData['x']), average='micro')
            else:
                return f1_score(self.dataRepo.testingData['y'], self.skModel.predict(self.dataRepo.testingData['x']), average='micro')

    def getAccurracy(self, *args, **kwargs):

        if self.skModel is None:
            raise SystemError('INVALID MODEL, PLEASE TRAIN')
        if 'data' in kwargs:
            if kwargs['data'] == "training":
                return accuracy_score(self.dataRepo.trainingData['y'], self.skModel.predict(self.dataRepo.trainingData['x']))
            else:
                return accuracy_score(self.dataRepo.testingData['y'], self.skModel.predict(self.dataRepo.testingData['x']))



    def debugPrintMetrics(self, *args, **kwargs):

        if self.skModel is None:
            raise SystemError('INVALID MODEL, PLEASE TRAIN')

        # Test accuracy
        print("\nThe test accuracy is: ")
        print("training data: " + str(self.getAccurracy(data='training')))
        print("testing data: " + str(self.getAccurracy(data='testing')))

        # Test f1 score
        print("\nThe test f1 score is: ")
        print("training data: " + str(self.getF1Score(data='training')))
        print("testing data: " + str(self.getF1Score(data='testing')))
