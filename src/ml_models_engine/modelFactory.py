from src.enums.smells import SystemSmells
from src.enums.models import SystemModels
from src.ml_models.DecisionTreeModel import DecisionTree
from src.ml_models.RandomForestModel import RandomForest
# from src.ml_models.DecisionTreeModel import DecisionTree
# from src.ml_models.DecisionTreeModel import DecisionTree

class ModelFactory():


    def createModel(self, model, smell, *args, **kwargs):
        """ Creates the model to work with. Receives the model name and the enum of the smell"""

        if(model == SystemModels.DECISION_TREE):
            return DecisionTree(smell=smell, *args, **kwargs)
        elif(model == SystemModels.RANDOM_FOREST):
            return RandomForest(smell=smell, *args, **kwargs)
        elif(model == SystemModels.NAIVE_BAYES):
            return DecisionTree(smell=smell, *args, **kwargs)
        elif(model == SystemModels.SVC_LINEAR):
            return DecisionTree(smell=smell, *args, **kwargs)
        elif(model == SystemModels.SVC_POLYNOMIAL):
            return DecisionTree(smell=smell, *args, **kwargs)
        elif(model == SystemModels.SVC_RBF):
            return DecisionTree(smell=smell, *args, **kwargs)
        elif(model == SystemModels.SVC_SIGMOID):
            return DecisionTree(smell=smell, *args, **kwargs)