import json
import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer

class FAISSIndex:
    def __init__(self, kb_file='kb_labelled.json'):
        with open(kb_file, 'r') as f:
            self.kb = json.load(f)
        self.vectorizer = TfidfVectorizer()
        self.texts = [entry['text'] for entry in self.kb]
        self.embeddings = self.vectorizer.fit_transform(self.texts).toarray()
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(np.array(self.embeddings, dtype='float32'))

    def search(self, query, top_k=5):
        query_vec = self.vectorizer.transform([query]).toarray().astype('float32')
        distances, indices = self.index.search(query_vec, top_k)
        return [(self.kb[i]['text'], distances[0][idx]) for idx, i in enumerate(indices[0])]