# examples

## Installation

You can install the necessary libraries to run the examples using
[poetry](https://python-poetry.org).

### Install poetry

To install poetry, it is recommended to run:

In linux / osx / bashonwindows: 

```
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
```

In windows:

```
(Invoke-WebRequest -Uri https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py -UseBasicParsing).Content | python -
```

Alternatively, it can be installed using `pip` (not recommended):

```
pip install poetry
```

See the official [documentation](https://python-poetry.org/docs/) for more
information regarding poetry installation

### Install dependencies

To install dependencies, simply run:

```
poetry install
```

## Running the example

Once the dependencies are fullfiled, the example can be executed using:

```
poetry run python main.py
```
