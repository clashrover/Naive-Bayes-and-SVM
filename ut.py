import json
import sys
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from functools import lru_cache

__author__= 'KD'


def json_writer(data, fname):
    """
        Write multiple json files
        Args:
            data: list(dict): list of dictionaries to be written as json
            fname: str: output file name
    """
    with open(fname, mode="w") as fp:
        for line in data:
            json.dump(line, fp)
            fp.write("\n")


def json_reader(fname):
    """
        Read multiple json files
        Args:
            fname: str: input file
        Returns:
            generator: iterator over documents 
    """
    for line in open(fname, mode="r"):
        yield json.loads(line)


def _stem(doc, stem_fn, en_stop, return_tokens):
    tokens = word_tokenize(doc.lower())
    stopped_tokens = filter(lambda token: token not in en_stop, tokens)
    stemmed_tokens = map(lambda token: stem_fn(token), stopped_tokens)
    if not return_tokens:
        return ' '.join(stemmed_tokens)
    return list(stemmed_tokens)

# ps = PorterStemmer()
# ps_stem = lru_cache(maxsize=None)(ps.stem) # use this function to stem
sb = SnowballStemmer('english')
sb_stem = lru_cache(maxsize=None)(sb.stem)
# ls = LancasterStemmer()
# ls_stem = lru_cache(maxsize=None)(ls.stem)

def getStemmedDocuments(docs, return_tokens=True):
    """
        Args:
            docs: str/list(str): document or list of documents that need to be processed
            returpythn_tokens: bool: return a re-joined string or tokens
        Returns:
            str/list(str): processed document or list of processed documents
        Example: 
            new_text = "It is important to by very pythonly while you are pythoning with python.
                All pythoners have pythoned poorly at least once."
            print(getStemmedDocuments(new_text))
        Reference: https://pythonprogramming.net/stemming-nltk-tutorial/
    """
    en_stop = set(stopwords.words('english'))
    
    if isinstance(docs, list):
        output_docs = []
        for item in docs:
            output_docs.append(_stem(item, sb_stem, en_stop, return_tokens))
        return output_docs
    else:
        return _stem(docs, sb_stem, en_stop, return_tokens)
