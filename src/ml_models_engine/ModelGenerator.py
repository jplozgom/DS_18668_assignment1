from src.ml_models.DecisionTreeModel import DecisionTree

class ModelGenerator():

    """Class in charge of training a model and storing the resulting mo del in memory"""

    def trainModel(self, smell, model):
        # --1. create model
        model = DecisionTree(smell=smell, model=model)
        # --2. load data set instance
        model.loadTrainingAndTestingData();
        #3. train the model
        model.trainModel()

