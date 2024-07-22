import faiss
import numpy as np

class DocumentStore:
    def __init__(self):
        self.documents = []
        self.embeddings = None
        self.index = None

    def add_documents(self, docs, embeddings):
        self.documents.extend(docs)
        embeddings = np.vstack(embeddings)
        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack((self.embeddings, embeddings))
        
        if self.index is None:
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)

    def search(self, query_embedding, top_k=5):
        D, I = self.index.search(query_embedding, top_k)
        return [self.documents[i] for i in I[0]]
