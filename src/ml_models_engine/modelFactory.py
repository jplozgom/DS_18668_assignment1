from src.enums.smells import SystemSmells
from src.enums.models import SystemModels
from src.ml_models.DecisionTreeModel import DecisionTree
from src.ml_models.RandomForestModel import RandomForest
from src.ml_models.NaiveBayesModel import NaiveBayes
from src.ml_models.svc import SupportVectorClassifier
# from src.ml_models.DecisionTreeModel import DecisionTree
# from src.ml_models.DecisionTreeModel import DecisionTree

class ModelFactory():


    def createModel(self, model, smell, debug, *args, **kwargs):
        """ Creates the model to work with. Receives the model name and the enum of the smell"""

        if(model == SystemModels.DECISION_TREE):
            return DecisionTree(smell=smell, debug=debug)
        elif(model == SystemModels.RANDOM_FOREST):
            return RandomForest(smell=smell, debug=debug)
        elif(model == SystemModels.NAIVE_BAYES):
            return NaiveBayes(smell=smell, debug=debug)
        elif(model == SystemModels.SVC_LINEAR):
            return SupportVectorClassifier(smell=smell, debug=debug, kernel="linear", model=model)
        elif(model == SystemModels.SVC_POLYNOMIAL):
            return SupportVectorClassifier(smell=smell, debug=debug, kernel="poly", model=model)
        elif(model == SystemModels.SVC_RBF):
            return SupportVectorClassifier(smell=smell, debug=debug, kernel="rbf", model=model)
        elif(model == SystemModels.SVC_SIGMOID):
            return SupportVectorClassifier(smell=smell, debug=debug, kernel="sigmoid", model=model)