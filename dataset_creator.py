import json
from datasets import Dataset
from data_processor import DataProcessor

class DatasetCreator:
    def __init__(self):
        self.data_processor = DataProcessor()
        self.dataset_file = "train_dataset.json"
        self.dataset = self.load_dataset()

    def create_initial_dataset(self, pdf_files, urls, questions):
        texts = self.data_processor.process_data(pdf_files, urls)
        for question in questions:
            context = self.find_relevant_context(question["question"], texts)
            self.dataset.append({"question": question["question"], "context": context})
        self.save_dataset()

    def find_relevant_context(self, question, texts):
        for text in texts:
            if question.lower() in text.lower():
                return text
        return "Context not found."

    def save_dataset(self):
        with open(self.dataset_file, "w") as f:
            json.dump(self.dataset, f, indent=4)

    def load_dataset(self):
        try:
            with open(self.dataset_file, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return []

    def update_dataset(self, question, context):
        self.dataset.append({"question": question, "context": context})
        self.save_dataset()
