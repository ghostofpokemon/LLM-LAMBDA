# llm-lambda

[![PyPI](https://img.shields.io/pypi/v/llm-lambda.svg)](https://pypi.org/project/llm-lambda/0.1.1)
[![GitHub release (latest by date including pre-releases)](https://img.shields.io/github/v/release/ghostofpokemon/llm-lambda?include_prereleases)](https://github.com/ghostofpokemon/llm-lambda/releases)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/ghostofpokemon/llm-lambda/blob/main/LICENSE)

LLM access to Hermes3 models by lambda Labs

## Installation

Install this plugin in the same environment as [LLM](https://llm.datasette.io/).

```bash
llm install llm-lambda
```

## Usage

First, set an [API key](https://cloud.lambdalabs.com/api-keys) for Lambda Labs:

```bash
llm keys set lambda
# Paste key here
```

Run `llm models` to list the models, and `llm models --options` to include a list of their options.

Run prompts like this:

```bash
llm -m h3-chat "Describe a Numogram ritual in Lemuria"
llm -m h3-completion "The time-sorcerer activated the Barbelith, causing"
llm -m h3-128-chat "Explain the concept of techno-shamanism"
llm -m h3-128-completion "In the year 2099, the first human-AI hybrid"
```

## Development

To set up this plugin locally, first checkout the code. Then create a new virtual environment:

```bash
cd llm-lambda
python3 -m venv venv
source venv/bin/activate
```

Now install the dependencies and test dependencies:

```bash
llm install -e '.[test]'
```
