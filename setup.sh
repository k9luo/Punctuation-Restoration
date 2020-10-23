mkdir -p data/external/embeddings
wget http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove*.zip
mv glove* data/external/embeddings

conda env create -f environment.yml
python -m ipykernel install --user --name=restore_punct
