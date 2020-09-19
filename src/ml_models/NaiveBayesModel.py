#Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from src.ml_models.mlModel import MLModel
from src.enums.models import SystemModels
from sklearn.model_selection import cross_val_score
from sklearn.metrics import SCORERS as sco
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import numpy as np



class NaiveBayes(MLModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model = SystemModels.NAIVE_BAYES
        self.convertYToInt = True

    def trainModel(self, *args, **kwargs):

        modelClassifier = GaussianNB()
        trainingData = self.dataRepo.trainingData
        testingData = self.dataRepo.testingData

        # Train the model using the training sets
        modelClassifier.fit(trainingData['x'], trainingData['y'])
        self.skModel = modelClassifier
        if self.debug:
            self.debugPrintMetrics()
        self.saveModel()