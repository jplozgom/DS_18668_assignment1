from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from src.ml_models.mlModel import MLModel
from src.enums.models import SystemModels
from sklearn.model_selection import cross_val_score
from sklearn.metrics import SCORERS as sco
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import numpy as np



class DecisionTree(MLModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model = SystemModels.DECISION_TREE
        # num of leafs to look for in the grid search
        self.num_leafs = [1, 5, 10, 20, 50, 100]
        # depth of tree
        self.depths = np.arange(1, 21)
        # num folds
        self.cv = 10

    def trainModel(self, *args, **kwargs):

        """ trains a model for a decision forest """

        treeClassifier = DecisionTreeClassifier()
        trainingData = self.dataRepo.trainingData
        testingData = self.dataRepo.testingData

        if self.useGridSearch:
            gridParams = { 'criterion':['gini','entropy'],'max_depth': self.depths, 'min_samples_leaf': self.num_leafs}
            gridSearch = GridSearchCV(treeClassifier, gridParams, cv=self.cv, scoring="accuracy", return_train_score=True)
            gridSearch.fit(trainingData['x'], trainingData['y'])
            self.skModel = gridSearch.best_estimator_
            if self.debug:
                self.debugPrintMetrics()
            self.saveModel()
        elif self.useRandomSearch:
            # TODO for next iteration JP
            pass
        else:
            cross_val_score(treeClassifier, trainingData['x'], trainingData['y'], cv=self.cv, scoring="accuracy")
            treeClassifier.fit(trainingData['x'], trainingData['y'])
            self.skModel = treeClassifier
            if self.debug:
                self.debugPrintMetrics()
            self.saveModel()