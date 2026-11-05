#!/bin/bash

ENV_NAME="gnn_env"

echo "Starting stuff and installing libraries"

eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

# checks for the enviroment
if pyenv versions --bare | grep -qx "$ENV_NAME"; then
    echo "Found existing env: $ENV_NAME"
else 
    echo "Creating env: $ENV_NAME"
    pyenv virtualenv "$ENV_NAME"
fi 

echo "Activating python env"
pyenv activate "$ENV_NAME"

pip install -r requirements.txt
