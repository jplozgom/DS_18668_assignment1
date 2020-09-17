from src.enums.smells import SystemSmells
from src.enums.models import SystemModels
from src.ml_models_engine.ModelGenerator import ModelGenerator

class SmellController():

    """Class with access to the list of smells"""

    def getListOfSmells(self):
        """ Gets the List of smells"""
        return [listItem for listItem in SystemSmells]






class ModelController():

    def getListOfModels(self):
        """ Gets the List of models"""
        return [listItem for listItem in SystemModels]


    def trainModel(self, smellName, modelName):

        """Method that trains a model following the instructions from the upper level (CLI or GUI)"""

        modelGenerator = ModelGenerator()
        smell = SystemSmells[str(smellName).upper().replace(" ", "_")]
        model = SystemModels[str(modelName).upper().replace(" ", "_")]
        if smell != None and model != None:
            modelGenerator.trainModel(smell, model)
        else:
            raise "Invalid smell or model"




