from src.enums.smells import SystemSmells
from src.enums.models import SystemModels
from src.ml_models_engine.ModelGenerator import ModelGenerator
from src.ml_models_engine.ModelEvaluator import ModelEvaluator
from tabulate import tabulate
import click

class SmellController():

    """Class with access to the list of smells"""

    def getListOfSmells(self):
        """ Gets the List of smells"""
        return [listItem for listItem in SystemSmells]


class ModelController():

    def getListOfModels(self):
        """ Gets the List of models"""
        return [listItem for listItem in SystemModels]


    def trainModel(self, debugMode, *args, **kwargs):

        """Method that trains a model following the instructions from the upper level (CLI or GUI)"""

        modelGenerator = ModelGenerator()
        smells = self.__gatherSmellsFromInput(**kwargs)
        model = self.__gatherModelFromInput(**kwargs)

        if len(smells) >= 1 and model != None:
            for smell in smells:
                modelGenerator.trainModel(smell, model, debugMode)
                click.echo()
                click.echo(  click.style("Model trained successfully for "+smell.label()+". Now please do run.py evaluate to compare the metrics to other smells or run.py compare to see the metrics of your model with the training and testing set" , fg='green'))
                click.echo()
        else:
            raise ValueError("Invalid smell or model")


    def evaluateModelForSmells(self, *args, **kwargs):

        """Method that evaluate a model used in one or more smells following the input from the upper level (CLI or GUI)"""

        modelEvaluator = ModelEvaluator()
        smells = self.__gatherSmellsFromInput(**kwargs)
        model = self.__gatherModelFromInput(**kwargs)

        if len(smells) >= 0 and model != None:
            responseData = []
            responseData = modelEvaluator.evaluateModel(smells, model)
            click.echo()
            click.echo(  click.style("Evaluation of a " + model.label() + " used to predict the smell following " , fg='green'))
            click.echo()
            click.echo(tabulate(responseData, headers=["Smell", "Accuracy", "F1-score"]))
            click.echo()
        else:
            raise ValueError("Invalid smell or model")


    def evaluateModelMetrics(self, *args, **kwargs):

        """Method that compares the metrics of a model for a specific smells with both training and testing data following the input from the upper level (CLI or GUI)"""

        modelEvaluator = ModelEvaluator()
        smells = self.__gatherSmellsFromInput(**kwargs)
        model = self.__gatherModelFromInput(**kwargs)

        if len(smells) > 0 and model != None:
            click.echo("")
            click.echo(  click.style("Evaluation of a " + model.label() + " used to predit the following Smells " , fg='green'))
            click.echo("")
            for smell in smells:
                responseData = modelEvaluator.compareScore(smell, model)
                click.echo(tabulate(responseData, headers="firstrow"))
                click.echo("")
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
        if "smell" in kwargs and type(kwargs['smell']) == str:
            smellName = str(kwargs['smell'])
            smellKey = str(smellName).upper().replace(" ", "_")
            if smellKey in SystemSmells.__members__:
                smells.append(SystemSmells[smellKey])
            elif smellKey == "ALL" :
                [smells.append(e) for e in SystemSmells]
            else:
                raise ValueError("Invalid smell '" + smellName+"'")

        # gather many smell
        elif "smell" in kwargs and type(kwargs['smell']) == tuple:
            for smellName in kwargs['smell']:
                smellKey = str(smellName).upper().replace(" ", "_")
                if smellKey in SystemSmells.__members__:
                    smells.append(SystemSmells[smellKey])
                elif smellKey == "ALL" :
                    [smells.append(e) for e in SystemSmells]
                else:
                    raise ValueError("Invalid smell '" + smellName+"'")


        return smells