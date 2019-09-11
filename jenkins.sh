#Make sure gcc is installed 
conda --version
pip --version
# Cleanup old env (if any)
conda env remove --name ampligraph || true

# Create & activate Virtual Environment
conda create --name ampligraph python=3.6
source activate ampligraph

# Install library
if [[ $# -eq 0 ]] ; then
    echo "install tensorflow CPU mode"
    pip install tensorflow==1.13.1
else 
    if [[ $1 == "gpu" ]] ; then
        echo "install tensorflow GPU mode"
        conda install cudatoolkit=10.0
        conda install cudnn=7.6
        pip install tensorflow-gpu==1.13.1 --no-cache-dir
    fi
fi

pip install . -v

# configure dataset location
export AMPLIGRAPH_DATA_HOME=/var/datasets

# run flake8 linter
flake8 ampligraph --max-line-length 120 --ignore=W291,W293 # ignoring some white space related errors

# run unit tests
pytest tests

# build documentation
cd docs
make clean autogen html

# cleanup: remove conda env
source deactivate
conda env remove --name ampligraph
