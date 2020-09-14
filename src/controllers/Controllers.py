from src.enums.smells import Smells
from src.enums.models import Models

class SmellController():

    """Class with access to the list of smells"""

    def getListOfSmells(self):
        """ Gets the List of smells"""
        return [listItem for listItem in Smells]






class ModelController():

    def getListOfModels(self):
        """ Gets the List of models"""
        return [listItem for listItem in Models]


    def trainModel(self, modelName, smellsNames):

        """Method that trains a model following the instructions from the upper level (CLI or GUI)"""

        pass

