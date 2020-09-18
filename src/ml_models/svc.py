#Import Gaussian Naive Bayes model
from sklearn import svm
from sklearn.model_selection import cross_val_score
from src.ml_models.mlModel import MLModel
from src.enums.models import SystemModels
from sklearn.model_selection import cross_val_score
from sklearn.metrics import SCORERS as sco
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import numpy as np



class SupportVectorClassifier(MLModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model = SystemModels.NAIVE_BAYES
        self.convertYToInt = True
        self.kernel = 'linear'

    def trainModel(self, *args, **kwargs):

        modelClassifier = svm.SVC(kernel=self.kernel) # Linear Kernel
        trainingData = self.dataRepo.trainingData
        testingData = self.dataRepo.testingData

        if self.useGridSearch and False:
            pass
            # gridParams = { 'criterion':['gini','entropy'],'max_depth': self.depths, 'min_samples_leaf': self.num_leafs}
            # gridSearch = GridSearchCV(modelClassifier, gridParams, cv=self.cv, scoring="accuracy", return_train_score=True)
            # gridSearch.fit(trainingData['x'], trainingData['y'])
            # self.skModel = gridSearch.best_estimator_
            # self.debugPrintMetrics()
            # self.saveModel()
        elif self.useRandomSearch and False:
            # TODO
            pass
        else:
            # for now go to default hyper parameters
            # Train the model using the training sets
            modelClassifier.fit(trainingData['x'], trainingData['y'])
            self.skModel = modelClassifier
            self.debugPrintMetrics()
            self.saveModel()