import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import click

# initialice the group that will have all of our commands
@click.group()
def cli():
    pass

# Command 1
@cli.command()
def initdb():
    click.echo('Initialized the database')


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