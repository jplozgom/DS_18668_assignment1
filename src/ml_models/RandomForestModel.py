from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from src.ml_models.mlModel import MLModel
from src.enums.models import SystemModels
from sklearn.model_selection import cross_val_score
from sklearn.metrics import SCORERS as sco
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import numpy as np



class RandomForest(MLModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model = SystemModels.RANDOM_FOREST
        #Number of trees in the forest --- generate ten numbers between 10 and 70
        self.nEstimators = [int(x) for x in np.linspace(start=10, stop=70, num=10)]
        #max features or number of features to consider in every split
        self.maxFeatures = ['auto', 'sqrt']
        # levels of depth in the tree
        self.maxDepth = [2,4]
        # number of samples needed to split a node
        self.minSamplesSplit = [2,5]
        # number of samples required in each leaf node
        self.minSamplesLeaf = [1,2]

        self.boostrap = [True, False]
        # num folds
        self.cv = 10



    def trainModel(self, *args, **kwargs):

        """ trains a model with for a random forest """

        modelClassifier = RandomForestClassifier()
        trainingData = self.dataRepo.trainingData
        testingData = self.dataRepo.testingData

        if self.useGridSearch:
            gridParams = {
                'n_estimators': self.nEstimators,
                'max_features': self.maxFeatures,
                'max_depth': self.maxDepth,
                'min_samples_split': self.minSamplesSplit,
                'min_samples_leaf': self.minSamplesLeaf,
                'bootstrap': self.boostrap
            }
            gridSearch = GridSearchCV(estimator=modelClassifier, param_grid=gridParams, cv=self.cv, scoring="accuracy", return_train_score=True, verbose=0, n_jobs=4)
            gridSearch.fit(trainingData['x'], trainingData['y'])
            self.skModel = gridSearch.best_estimator_
            if self.debug:
                self.debugPrintMetrics()
            self.saveModel()
        elif self.useRandomSearch:
            # TODO for next iteration JP
            pass
        else:
            cross_val_score(modelClassifier, trainingData['x'], trainingData['y'], cv=self.cv, scoring="accuracy")
            modelClassifier.fit(trainingData['x'], trainingData['y'])
            self.skModel = modelClassifier
            if self.debug:
                self.debugPrintMetrics()
            self.saveModel()