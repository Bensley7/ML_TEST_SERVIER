import click

from molecular_ml.train import train_model
from molecular_ml.evaluate import evaluate_model
from molecular_ml.predict import predict_model
from molecular_ml.web.server import run_app


@click.command()
@click.option(
    "--train_data_path",
    required=True,
    help="train path data",
    type=click.Path(exists=True, dir_okay=False, readable=True),
)
@click.option(
    "--val_data_path",
    help="validation path data",
    type=click.Path(exists=True, dir_okay=False, readable=True),
)
@click.option(
    "--test_data_path",
    help="test path data",
    type=click.Path(exists=True, dir_okay=False, readable=True),
)
@click.option(
    "--config_file",
    default="",
    help="path to config file",
    type=str,
)
@click.option(
    "--model_dir",
    default="mol_models/",
    help="directory to save the model",
    type=str,
)
@click.option(
    "--model_name",
    default="model1",
    help="name of the model",
    type=str,
)
def train(
    train_data_path: str,
    val_data_path: str,
    test_data_path: str,
    config_file: str,
    model_dir: str,
    model_name: str,
):
    train_model(train_data_path,
                val_data_path,
                test_data_path,
                config_file=config_file,
                model_dir=model_dir,
                model_name=model_name)

@click.command()
@click.option(
    "--model_path",
    required=True,
    help="weight path of the model",
    type=click.Path(exists=True, dir_okay=False, readable=True),
)
@click.option(
    "--smile",
    required=True,
    help="smile of a molecule",
    type=str,
)
@click.option(
    "--config_file",
    default="",
    help="path to config file",
    type=str,
)

def predict(model_path: str, smile: str, config_file: str):
    predict_model(model_path, smile, config_file=config_file)


@click.command()
@click.option(
    "--model_path",
    required=True,
    help="weight path of the model",
    type=click.Path(exists=True, dir_okay=False, readable=True),
)
@click.option(
    "--val_data_path",
    required=True,
    help="Evaluation csv with ground truth path",
    type=str,
)
@click.option(
    "--config_file",
    required=False,
    default="",
    help="path to config file",
    type=str,
)

def evaluate(model_path: str, val_data_path: str, config_file=str):
    evaluate_model(model_path, val_data_path, config_file=config_file)


@click.command()
@click.option(
    "--model_path",
    required=True,
    help="weight path of the model",
    type=click.Path(exists=True, dir_okay=False, readable=True),
)
@click.option(
    "--config_file",
    default="",
    help="path to config file",
    type=str,
)
@click.option(
    "--host",
    default='0.0.0.0',
    help="host name",
    type=str,
)
@click.option(
    "--port",
    default=8000,
    help="port name",
    type=int,
)

def web(model_path: str, config_file: str, host: str, port: int):
    run_app(model_path,
            config_file=config_file,
            host=host,
            port=port
            )


@click.group()
def run():
    pass


run.add_command(train)
run.add_command(evaluate)
run.add_command(predict)
run.add_command(web)
