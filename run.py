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

# Command 1 - Train a model for a smell
@cli.command(name='train')
# @click.argument('schedule_csv', type=click.Path(exists=True, readable=True))
# @click.option('--schedule_csv', help='Filepath to schedule csv file', type=click.Path(exists=True, readable=True),required=True)

@click.option('--smell', help='Smell used to train a Model.', required=True, type=click.STRING)
@click.option('--model', help='Model used to predict a smell from a dataset.', required=True, type=click.STRING)
def trainModel(smell, model):
    """Trains a model for a smell."""
    click.echo('Trains a model for a smell' + " - " + smell + " - " + model)


# Command 2 - Predict a smell using a
@cli.command(name='predict')
@click.option('--smell', help='Smell used to train a Model.')
@click.option('--model', help='Model used to predict a smell from a dataset.')
def predictSmell():
    """Predicts a smell using a model generated previously"""

    click.echo('Predicts a smell using a model generated previously')

# Command 2
@cli.command()
@click.option('--count', default=1, help='Number of greetings.')
@click.option('--name', prompt='Your name', help='The person to greet.')
def hello(count, name):
    """Simple program that greets NAME for a total of COUNT times."""
    for x in range(count):
        click.echo('Hello %s!' % name)




# main function to call
if __name__ == '__main__':
    cli()