#!/bin/bash
HOME_DIR="."

cd $HOME_DIR
python -m venv venv
source venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --no-cache-dir
pip install tqdm pandas matplotlib seaborn --no-cache-dir
pip install -U scikit-learn --no-cache-dir

pip install 
pip install -e .