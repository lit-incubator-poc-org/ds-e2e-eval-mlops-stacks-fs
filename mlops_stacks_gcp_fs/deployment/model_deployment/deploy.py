import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).parent.parent.parent.resolve()))
from utils import get_deployed_model_alias_for_env
from mlflow.tracking import MlflowClient


def deploy(model_uri, env):
    """
    Deploys an already-registered model by setting an appropriate alias for Unity Catalog model deployment.

    :param model_uri: URI of the model to deploy. Must be in the format "models:/<name>/<version-id>", as described in
                      https://www.mlflow.org/docs/latest/model-registry.html#fetching-an-mlflow-model-from-the-model-registry
    :param env: name of the environment in which we're performing deployment, i.e one of "dev", "staging", "prod".
                Defaults to "dev"
    :return:
    """
    _, model_name, version = model_uri.split("/")
    client = MlflowClient()
    
    # For Unity Catalog, use aliases instead of stages
    target_alias = get_deployed_model_alias_for_env(env)
    
    # Set the alias for the model version
    client.set_registered_model_alias(
        name=model_name,
        alias=target_alias,
        version=version
    )
    print(f"Successfully deployed model with URI {model_uri} to {env} using alias '{target_alias}'")


if __name__ == "__main__":
    deploy(model_uri=sys.argv[1], env=sys.argv[2])
