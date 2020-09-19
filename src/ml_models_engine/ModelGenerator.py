from src.ml_models_engine.modelFactory import ModelFactory

class ModelGenerator():

    """Class in charge of training a model and storing the resulting mo del in memory"""

    def trainModel(self, smell, model, debug, *args, **kwargs):
        # --1. create model
        mlModel = ModelFactory().createModel(model, smell, debug, *args, **kwargs)
        # --2. load data set instance
        mlModel.loadTrainingAndTestingData();
        #3. train the mlModel
        mlModel.trainModel()

