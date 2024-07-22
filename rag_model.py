from transformers import RagTokenizer, RagTokenForGeneration

class RAGModel:
    def __init__(self, model_name="facebook/rag-token-base", document_store=None):
        self.tokenizer = RagTokenizer.from_pretrained(model_name)
        self.model = RagTokenForGeneration.from_pretrained(model_name)
        self.document_store = document_store
    
    def generate(self, query):
        inputs = self.tokenizer(query, return_tensors="pt")
        query_embedding = self.tokenizer(query, return_tensors="pt", padding=True, truncation=True)['input_ids']
        query_embedding = self.model.rag.retriever.question_encoder(query_embedding)[0].detach().cpu().numpy()
        context = self.document_store.search(query_embedding)
        context_str = ' '.join(context)
        context_inputs = self.tokenizer(context_str, return_tensors="pt", padding=True, truncation=True)
        generated = self.model.generate(input_ids=inputs["input_ids"], context_input_ids=context_inputs["input_ids"], num_return_sequences=1)
        return self.tokenizer.decode(generated[0], skip_special_tokens=True)
