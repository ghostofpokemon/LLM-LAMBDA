import llm
from llm.default_plugins.openai_models import Chat, Completion
from pathlib import Path
import json
import time
import httpx

def get_lambda_models(key):
    headers = {"Authorization": f"Bearer {key}"}
    return fetch_cached_json(
        url="https://api.lambdalabs.com/v1/models",
        path=llm.user_dir() / "lambda_models.json",
        cache_timeout=3600,
        headers=headers,
    )["data"]

def get_model_ids_with_aliases(models):
    model_aliases = {
        "hermes-3-llama-3.1-405b-fp8": ["h3-chat", "h3-completion"],
        "hermes-3-llama-3.1-405b-fp8-128k": ["h3-128-chat", "h3-128-completion"],
    }

    models_with_aliases = []
    for model in models:
        model_id = model["id"]
        aliases = model_aliases.get(model_id, [])
        models_with_aliases.append((model_id, aliases))

    return models_with_aliases

class LambdaChat(Chat):
    needs_key = "lambda"
    key_env_var = "LLM_LAMBDA_KEY"

    def __str__(self):
        return f"Lambda Chat: {self.model_id}"

class LambdaCompletion(Completion):
    needs_key = "lambda"
    key_env_var = "LLM_LAMBDA_KEY"

    def __str__(self):
        return f"Lambda Completion: {self.model_id}"

@llm.hookimpl
def register_models(register):
    key = llm.get_key("", "lambda", "LLM_LAMBDA_KEY")
    if not key:
        return
    try:
        models = get_lambda_models(key)
        models_with_aliases = get_model_ids_with_aliases(models)
        for model_id, aliases in models_with_aliases:
            chat_aliases = [alias for alias in aliases if alias.endswith("-chat")]
            completion_aliases = [alias for alias in aliases if alias.endswith("-completion")]

            register(
                LambdaChat(
                    model_id=f"lambdachat/{model_id}",
                    model_name=model_id,
                    api_base="https://api.lambdalabs.com/v1",
                ),
                aliases=chat_aliases
            )
            register(
                LambdaCompletion(
                    model_id=f"lambdacompletion/{model_id}",
                    model_name=model_id,
                    api_base="https://api.lambdalabs.com/v1",
                ),
                aliases=completion_aliases
            )
    except DownloadError as e:
        print(f"Error fetching Lambda models: {e}")

class DownloadError(Exception):
    pass

def fetch_cached_json(url, path, cache_timeout, headers=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.is_file() and time.time() - path.stat().st_mtime < cache_timeout:
        with open(path, "r") as file:
            return json.load(file)

    try:
        response = httpx.get(url, headers=headers, follow_redirects=True)
        response.raise_for_status()
        with open(path, "w") as file:
            json.dump(response.json(), file)
        return response.json()
    except httpx.HTTPError:
        if path.is_file():
            with open(path, "r") as file:
                return json.load(file)
        else:
            raise DownloadError(f"Failed to download data and no cache is available at {path}")

@llm.hookimpl
def register_commands(cli):
    @cli.command()
    def lambda_models():
        "List available Lambda models"
        key = llm.get_key("", "lambda", "LLM_LAMBDA_KEY")
        if not key:
            print("Lambda API key not set. Use 'llm keys set lambda' to set it.")
            return
        try:
            models = get_lambda_models(key)
            models_with_aliases = get_model_ids_with_aliases(models)
            for model_id, aliases in models_with_aliases:
                chat_aliases = [alias for alias in aliases if alias.endswith("-chat")]
                completion_aliases = [alias for alias in aliases if alias.endswith("-completion")]

                print(f"Lambda Chat: lambdachat/{model_id}")
                if chat_aliases:
                    print(f"  Aliases: {', '.join(chat_aliases)}")

                print(f"Lambda Completion: lambdacompletion/{model_id}")
                if completion_aliases:
                    print(f"  Aliases: {', '.join(completion_aliases)}")
                print()
        except DownloadError as e:
            print(f"Error fetching Lambda models: {e}")
