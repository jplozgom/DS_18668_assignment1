from src.ml_models_engine.modelFactory import ModelFactory

class ModelEvaluator():

    """Class in charge of evaluating the performance of previously trained models"""

    def evaluateModel(self, smells, model):
        responseData = []
        # --1. create model classes for each smell
        for smell in smells:
            mlModel = ModelFactory().createModel(model, smell, False)
            # --2. load data set instance
            mlModel.loadTrainingAndTestingData();
            # --3. load scikit model from disk
            mlModel.retrieveModelResults();

            if(mlModel.skModel != None):
                responseData.append(mlModel.getScoringData())

        return responseData


    def compareScore(self, smell, model):

        responseData = []
        # --1. create model classes for each smell
        mlModel = ModelFactory().createModel(model, smell, False)
        # --2. load data set instance
        mlModel.loadTrainingAndTestingData();
        # --3. load scikit model from disk
        mlModel.retrieveModelResults();

        if(mlModel.skModel != None):
            responseData = mlModel.getTestTrainingScoringData()

        return responseData

