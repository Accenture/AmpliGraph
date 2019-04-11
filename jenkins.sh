#Make sure gcc is installed 
conda --version
pip --version
# Cleanup old env (if any)
conda env remove --name ampligraph || true

# Create & activate Virtual Environment
conda create --name ampligraph python=3.6
source activate ampligraph

# Install library
conda install tensorflow-gpu==1.12.0

pip install . -v

# configure dataset location
export AMPLIGRAPH_DATA_HOME=/var/datasets

# run unit tests
pytest tests

# build documentation
cd docs
make clean autogen html

# cleanup: remove conda env
source deactivate
conda env remove --name ampligraph
