from src.enums.smells import SystemSmells
from src.enums.models import SystemModels
from src.ml_models_engine.ModelGenerator import ModelGenerator
from src.ml_models_engine.ModelEvaluator import ModelEvaluator

class SmellController():

    """Class with access to the list of smells"""

    def getListOfSmells(self):
        """ Gets the List of smells"""
        return [listItem for listItem in SystemSmells]






class ModelController():

    def getListOfModels(self):
        """ Gets the List of models"""
        return [listItem for listItem in SystemModels]


    def trainModel(self, *args, **kwargs):

        """Method that trains a model following the instructions from the upper level (CLI or GUI)"""

        modelGenerator = ModelGenerator()
        smells = self.__gatherSmellsFromInput(**kwargs)
        model = self.__gatherModelFromInput(**kwargs)

        if len(smells) == 1 and model != None:
            modelGenerator.trainModel(smells[0], model)
        else:
            raise ValueError("Invalid smell or model")


    def evaluateModelForSmells(self, *args, **kwargs):

        """Method that evaluate a model used in one or more smells following the input from the upper level (CLI or GUI)"""

        modelEvaluator = ModelEvaluator()
        smells = self.__gatherSmellsFromInput(**kwargs)
        model = self.__gatherModelFromInput(**kwargs)

        if len(smells) > 0 and model != None:
            modelEvaluator.evaluateModel(smells, model)
        else:
            raise ValueError("Invalid smell or model")


    def evaluateModelMetrics(self, *args, **kwargs):

        """Method that compares the metrics of a model for a specific smells with both training and testing data following the input from the upper level (CLI or GUI)"""

        modelEvaluator = ModelEvaluator()
        smells = self.__gatherSmellsFromInput(**kwargs)
        model = self.__gatherModelFromInput(**kwargs)

        if len(smells) > 0 and model != None:
            modelEvaluator.compareScore(smells[0], model)
        else:
            raise ValueError("Invalid smell or model")


    def __gatherModelFromInput(self, *args, **kwargs):

        """ gather the model entered by the user """

        modelKey = ''
        if "model" in kwargs:
            modelName = kwargs['model']
            modelKey = str(modelName).upper().replace(" ", "_")
            if modelKey in SystemModels.__members__:
                return SystemModels[modelKey]
            else:
                raise ValueError("Invalid model '" + modelName+"'")

        return None

    def __gatherSmellsFromInput(self, *args, **kwargs):

        """ gather the smell or smells entered by the user """

        smellKey = ''
        smells = []
        # gather one smell
        if "smell" in kwargs:
            smellName = kwargs['smell']
            smellKey = str(smellName).upper().replace(" ", "_")
            if smellKey in SystemSmells.__members__:
                smells.append(SystemSmells[smellKey])
            else:
                raise ValueError("Invalid smell '" + smellName+"'")

        # gather many smell
        elif "smells" in kwargs:
            for smellName in kwargs['smells']:
                smellKey = str(smellName).upper().replace(" ", "_")
                if smellKey in SystemSmells.__members__:
                    smells.append(SystemSmells[smellKey])
                else:
                    raise ValueError("Invalid smell '" + smellName+"'")

        return smells