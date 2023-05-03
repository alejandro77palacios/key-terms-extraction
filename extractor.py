from lxml import etree
from sklearn.feature_extraction.text import TfidfVectorizer

from news_processor import NewsProcessor


class Extractor:

    def __init__(self, xml_path):
        self.xml = etree.parse(xml_path)
        self.corpus = self.xml.getroot()[0]
        self.news_info = []
        self.tfidf_matrix = None
        self.feature_names = None

    def preprocess(self):
        for news in self.corpus:
            processor = NewsProcessor(news[1].text)
            processor.process()
            pair = news[0].text.strip(), ' '.join(processor.tokens)
            self.news_info.append(pair)

    def compute_tfidf(self):
        vectoriser = TfidfVectorizer()
        sentences = [sentence for title, sentence in self.news_info]
        self.tfidf_matrix = vectoriser.fit_transform(sentences)
        self.feature_names = vectoriser.get_feature_names_out()

    def compute_best_words(self, index_sentence, num_best=5):
        tfidf_scores = self.tfidf_matrix[index_sentence].toarray()[0]
        best_indices = tfidf_scores.argsort()[-num_best:]
        best_words = [self.feature_names[i] for i in best_indices[::-1]]
        return ' '.join(best_words)

    def show_results(self):
        for i, news in enumerate(self.news_info):
            print()
            print(news[0], end=':\n')
            print(self.compute_best_words(i))

    def main(self):
        self.preprocess()
        self.compute_tfidf()
        self.show_results()




