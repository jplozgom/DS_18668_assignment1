from src.ml_models_engine.modelFactory import ModelFactory

class ModelEvaluator():

    """Class in charge of evaluating the performance of previously trained models"""

    def evaluateModel(self, smells, model):
        mlModels = []
        # --1. create model classes for each smell
        for smell in smells:
            mlModel = ModelFactory().createModel(model, smell)
            # --2. load data set instance
            mlModel.loadTrainingAndTestingData();
            # --3. load scikit model from disk
            mlModel.retrieveModelResults();

            if(mlModel.skModel != None):
                print("model retrieved")
            mlModels.append(mlModel)
            mlModel.debugPrintMetrics()

            #3. train the model


    def compareScore(self, smell, model):

        print(smell)
        print(model)
        pass
