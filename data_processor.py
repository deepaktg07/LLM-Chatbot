import PyPDF2
import requests
from bs4 import BeautifulSoup

class DataProcessor:
    def extract_text_from_pdf(self, pdf_path):
        text = ""
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                text += reader.pages[page_num].extract_text()
        return text

    def extract_text_from_url(self, url):
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        return soup.get_text()

    def process_data(self, pdf_files, urls):
        texts = []
        for pdf in pdf_files:
            texts.append(self.extract_text_from_pdf(pdf))
        for url in urls:
            texts.append(self.extract_text_from_url(url))
        return texts
