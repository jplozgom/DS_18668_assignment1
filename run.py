import os, sys
from src.controllers.Controllers import *
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import click
import warnings
warnings.filterwarnings("ignore")

# initialice the group that will have all of our commands
@click.group()
def cli():
    pass


# Command 0 - List smells available
@cli.command(name='smells')
def listSmells():
    """Lists all the smells available for the system and the user to work with ."""
    click.echo("")

    for index, listItem in enumerate(SmellController().getListOfSmells()):
        click.echo(  click.style(str(index + 1) + ". " + listItem.label().upper() + ": ", fg='green') + listItem.description() )
    click.echo("")



# Command 1 - List models available
@cli.command(name='models')
def listModels():
    """Lists all the models available for the system and the user to work with and predict smells in data ."""
    click.echo("")

    for index, listItem in enumerate(ModelController().getListOfModels()):
        click.echo(  click.style(str(index + 1) + ". " + listItem.label().upper() + ": ", fg='green') + listItem.description() )
    click.echo("")

# Command 2 - Train a model for a smell
@cli.command(name='train')
@click.option('--smell', help='Smell used to train a Model.', required=True, type=click.STRING)
@click.option('--model', help='Model used to predict a smell from a dataset.', required=True, type=click.STRING)
@click.option('--debug', help='Debug mode to print the evaluation data the end.', is_flag=True)
def trainModel(smell, model, debug):
    """Trains a model for a smell."""
    modelController = ModelController();
    try:
        modelController.trainModel(debug, smell=smell, model=model)
    except Exception as e:
        click.echo('')
        click.echo(click.style(str(e), fg='red'))
        click.echo('')


# Command 3 - Evaluate the predictions of multiple smells a smell using a
@cli.command(name='evaluate')
@click.option('--smell', help='Smells the the user wants to evaluate with the selected model.', required=True, type=click.STRING, multiple=True)
@click.option('--model', help='Model used previously to train the a dataset. See the train command', required=True, type=click.STRING)
def evaluateModel(smell, model):
    """Evaluate a model for multiple smells using models generated previously"""
    modelController = ModelController();
    try:
        modelController.evaluateModelForSmells(smell=smell, model=model)
    except Exception as e:
        click.echo('')
        click.echo(click.style(str(e), fg='red'))
        click.echo('')


# Command 4 evaluate models
@cli.command(name='compare')
@click.option('--smell', help='Smell to evaluate (using it\'s training and test data).', required=True, type=click.STRING)
@click.option('--model', help='Model trained previously.', required=True, type=click.STRING)
def compareScores(smell, model):
    """Compare testing and training data metrics from a model generated for a smell"""
    modelController = ModelController();
    try:
        modelController.evaluateModelMetrics(smell=smell, model=model)
    except Exception as e:
        click.echo('')
        click.echo(click.style(str(e), fg='red'))
        click.echo('')


# main function to call
if __name__ == '__main__':
    cli()