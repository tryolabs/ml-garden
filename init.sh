#!/bin/bash

# Exit if a command fails
set -e

# Exit if project name or python version aren't specified as args
if [ -z "$1" ] || [ -z "$2" ]
then
    echo "Usage: $0 <project name> <python version to use>"
    exit
fi

# Sanitize project name
PACKAGE_NAME=$(echo "$1" | tr "-" "_")

# Install desired python version if not already installed
pyenv install --skip-existing $2

# Set python version for pyenv
pyenv local $2

# Initialize poetry env with specified python
poetry init --no-interaction --name $PACKAGE_NAME --python ^$2

# Make poetry use the correct version of python when spawning the venv
pyenv which python | xargs poetry env use

# Make poetry store the venv in the .venv folder inside the project
poetry config virtualenvs.in-project true --local

# Install venv and add dev deps
poetry add flake8 black isort --group dev

# Rename source code folder
mv project_base $PACKAGE_NAME

# If testing the template rename project imports in test script (for template development purposes)
# else remove the test script
if [ "$test" = true ]
then
    sed -i '' "s/project_base/$PACKAGE_NAME/" "$PACKAGE_NAME/test.py"
else
    rm "$PACKAGE_NAME/test.py"
fi

# Install current project's source
poetry install
