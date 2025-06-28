import re
from nltk.corpus import stopwords


def clean_str(s):
    s = re.sub(r"[^A-Za-z(),!?\'\`]", " ", s)
    s = re.sub(r"\'s", " is", s)
    s = re.sub(r"\'ve", " have", s)
    s = re.sub(r"\'t", " not", s)
    s = re.sub(r"\'re", " are", s)
    s = re.sub(r"\'d", " had", s)
    s = re.sub(r"\'ll", " will", s)
    s = re.sub(r",", " , ", s)
    s = re.sub(r"!", " ! ", s)
    s = re.sub(r"\(", " ", s)
    s = re.sub(r"\)", " ", s)
    s = re.sub(r"\?", " \? ", s)
    s = re.sub(r"\s{2,}", " ", s)
    s = re.sub(r'\S*(x{2,}|X{2,})\S*', "xx", s)
    s = re.sub(r'[^\x00-\x7F]+', "", s)
    tokens = s.split(' ')
    data = []
    words = stopwords.words('english')
    for token in tokens:
        if token not in words and len(token)>1:
            data.append(token)
    s = " ".join(data)
    return s.strip().lower()