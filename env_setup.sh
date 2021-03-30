conda create --prefix /pointer_env python=3.6
conda activate /pointer_env
conda install pytorch
pip install tqdm
pip install boto3
pip install requests
pip install regex
pip install nltk
bash requirement.sh
python -m spacy download en_core_web_sm
