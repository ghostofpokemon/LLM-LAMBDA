from click.testing import CliRunner
from llm.cli import cli
import json
import pytest
import llm_lambda

@pytest.mark.parametrize("set_key", (False, True))
def test_llm_models(set_key, user_path):
    runner = CliRunner()
    if set_key:
        (user_path / "keys.json").write_text(json.dumps({"lambda": "x"}), "utf-8")
    result = runner.invoke(cli, ["models", "list"])
    assert result.exit_code == 0, result.output
    fragments = (
        "Lambda Chat: lambdachat/hermes-3-llama-3.1-405b-fp8",
        "Lambda Completion: lambdacompletion/hermes-3-llama-3.1-405b-fp8",
        "Lambda Chat: lambdachat/hermes-3-llama-3.1-405b-fp8-128k",
        "Lambda Completion: lambdacompletion/hermes-3-llama-3.1-405b-fp8-128k",
    )
    for fragment in fragments:
        if set_key:
            assert fragment in result.output
        else:
            assert fragment not in result.output

@pytest.mark.parametrize("set_key", (False, True))
def test_lambda_models_command(set_key, user_path):
    runner = CliRunner()
    if set_key:
        (user_path / "keys.json").write_text(json.dumps({"lambda": "x"}), "utf-8")
    result = runner.invoke(cli, ["lambda-models"])
    assert result.exit_code == 0, result.output
    if set_key:
        assert "Lambda Chat: lambdachat/hermes-3-llama-3.1-405b-fp8" in result.output
        assert "Aliases: h3-chat" in result.output
        assert "Lambda Completion: lambdacompletion/hermes-3-llama-3.1-405b-fp8" in result.output
        assert "Aliases: h3-completion" in result.output
        assert "Lambda Chat: lambdachat/hermes-3-llama-3.1-405b-fp8-128k" in result.output
        assert "Aliases: h3-128-chat" in result.output
        assert "Lambda Completion: lambdacompletion/hermes-3-llama-3.1-405b-fp8-128k" in result.output
        assert "Aliases: h3-128-completion" in result.output
    else:
        assert "Lambda API key not set" in result.output
