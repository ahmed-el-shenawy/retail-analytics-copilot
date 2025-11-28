import os
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class DocumentRetriever:
    def __init__(self, docs_dir='docs/'):
        self.chunks = self._load_and_chunk(docs_dir)
        self._build_index()
    
    def _load_and_chunk(self, docs_dir):
        chunks = []
        for filename in os.listdir(docs_dir):
            if not filename.endswith('.md'):
                continue
            
            filepath = os.path.join(docs_dir, filename)
            with open(filepath, 'r') as f:
                content = f.read()
            
            # Split by sections (## headers)
            sections = content.split('\n## ')
            
            for i, section in enumerate(sections):
                if section.strip():
                    chunks.append({
                        'id': f'{filename}::chunk{i}',
                        'content': section.strip(),
                        'source': filename
                    })
        
        return chunks
    
    def _build_index(self):
        texts = [c['content'] for c in self.chunks]
        self.vectorizer = TfidfVectorizer(
            max_features=300,
            ngram_range=(1, 2),
            stop_words='english'
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
    
    def search(self, query, top_k=3):
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        top_indices = scores.argsort()[-top_k:][::-1]
        
        return [
            {**self.chunks[idx], 'score': float(scores[idx])}
            for idx in top_indices
        ]