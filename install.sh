#!/bin/bash

conda env update -f env.yml
pip install pre-commit black flake8 isort
pre-commit install
pre-commit autoupdate
