import gensim.downloader as api
import nltk

nltk.download('punkt')
nltk.download('wordnet')
v2w_model=api.load('glove-wiki-gigaword-100')
v2w_model.save(".//w2vecmodel.mod")
