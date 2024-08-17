import json
import llm
import pytest

DUMMY_MODELS = {
    "data": [
        {
            "id": "hermes-3-llama-3.1-405b-fp8",
            "name": "Hermes 3 Llama 3.1 405B FP8",
            "context_length": 4096,
        },
        {
            "id": "hermes-3-llama-3.1-405b-fp8-128k",
            "name": "Hermes 3 Llama 3.1 405B FP8 128K",
            "context_length": 131072,
        },
    ]
}

@pytest.fixture
def user_path(tmpdir):
    dir = tmpdir / "llm.datasette.io"
    dir.mkdir()
    return dir

@pytest.fixture(autouse=True)
def env_setup(monkeypatch, user_path):
    monkeypatch.setenv("LLM_USER_PATH", str(user_path))
    # Write out the models.json file
    (llm.user_dir() / "lambda_models.json").write_text(
        json.dumps(DUMMY_MODELS), "utf-8"
    )
