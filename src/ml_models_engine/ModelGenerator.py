from src.ml_models.DecisionTreeModel import DecisionTree
from src.enums.smells import SystemSmells
from src.enums.models import SystemModels

class ModelGenerator():

    """Class in charge of training a model and storing the resulting mo del in memory"""

    def trainModel(self, smell, model):

        if smell not in SystemSmells:
            # todo throw error
            return None
        if model not in SystemModels:
            # todo throw error
            return None
        # --1. create model
        model = DecisionTree(smell=smell)
        # --2. load data set instance
        model.loadTrainingAndTestingData();

        # print("testing data")
        # print(len(model.dataRepo.trainingData['x']))
        # print(len(model.dataRepo.trainingData['y']))
        # print("training data")
        # print(len(model.dataRepo.testingData['x']))
        # print(len(model.dataRepo.testingData['y']))
        # print(model.dataRepo.testingData['y'])

        #3. train the model
        model.trainModel()

