from string import punctuation

from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


class NewsProcessor:
    stop_words = stopwords.words('english')
    lemmatiser = WordNetLemmatizer()

    def __init__(self, text):
        self.text = text.strip().lower()
        self.tokens = None

    def tokenise(self):
        self.tokens = [t for t in word_tokenize(self.text) if not (t in punctuation or t in self.stop_words)]
        self.tokens.sort(reverse=True)

    def lematise(self):
        self.tokens = [self.lemmatiser.lemmatize(t, pos='n') for t in self.tokens]
        self.tokens = [t for t in self.tokens if len(t) > 1]

    def assign_pos_tags(self):
        pos_tags = pos_tag(self.tokens)
        self.tokens = [word for word, pos in pos_tags if pos == 'NN']

    def process(self):
        self.tokenise()
        self.lematise()
        self.assign_pos_tags()
