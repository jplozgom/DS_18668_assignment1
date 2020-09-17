import os, sys
from src.controllers.Controllers import *
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import click

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
def trainModel(smell, model):
    """Trains a model for a smell."""
    modelController = ModelController();
    modelController.trainModel(smell, model)


# Command 3 - Evaluate the predictions of multiple smells a smell using a
@cli.command(name='evaluate')
@click.option('--smells', help='Smells the user wnats to evaluate with the selected model.', required=True, type=click.STRING)
@click.option('--model', help='Model used previously to train the a dataset. See the train command', required=True, type=click.STRING)
def evaluateModel():
    """Predicts a smell using a model generated previously"""

    click.echo('Predicts a smell using a model generated previously')

# Command 4 evaluate models
@cli.command()
@click.option('--count', default=1, help='Number of greetings.')
@click.option('--name', prompt='Your name', help='The person to greet.')
def compareScores(count, name):
    """Simple program that greets NAME for a total of COUNT times."""
    for x in range(count):
        click.echo('Hello %s!' % name)




# main function to call
if __name__ == '__main__':
    cli()