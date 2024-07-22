import requests
from bs4 import BeautifulSoup
import json

# Function to scrape content from a URL
def scrape_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    return soup

# Function to extract questions and contexts from the Alkimi Exchange page
def extract_alkimi_exchange(soup):
    qna_pairs = []

    # Example extraction logic (this may need to be adapted based on actual page structure)
    for section in soup.find_all('section'):
        header = section.find(['h1', 'h2', 'h3'])
        if header:
            question = header.text.strip()
            context = ' '.join([p.text.strip() for p in section.find_all('p')])
            qna_pairs.append({"question": question, "context": context})

    return qna_pairs

# Function to extract questions and contexts from the How It Works page
def extract_how_it_works(soup):
    qna_pairs = []

    # Example extraction logic (this may need to be adapted based on actual page structure)
    for section in soup.find_all('section'):
        header = section.find(['h1', 'h2', 'h3'])
        if header:
            question = header.text.strip()
            context = ' '.join([p.text.strip() for p in section.find_all('p')])
            qna_pairs.append({"question": question, "context": context})

    return qna_pairs

# Function to extract questions and contexts from the Validators page
def extract_validators(soup):
    qna_pairs = []

    # Example extraction logic (this may need to be adapted based on actual page structure)
    for section in soup.find_all('section'):
        header = section.find(['h1', 'h2', 'h3'])
        if header:
            question = header.text.strip()
            context = ' '.join([p.text.strip() for p in section.find_all('p')])
            qna_pairs.append({"question": question, "context": context})

    return qna_pairs

# URLs to scrape
urls = [
    ('https://www.alkimi.org/tokenomics?section=alkimi-exchange', extract_alkimi_exchange),
    ('https://www.alkimi.org/how-it-works', extract_how_it_works),
    ('https://www.alkimi.org/tokenomics?section=validators', extract_validators)
]

# Load existing dataset
json_file_path = 'train_dataset.json'
try:
    with open(json_file_path, 'r') as f:
        dataset = json.load(f)
except FileNotFoundError:
    dataset = []

# Scrape and extract data
for url, extract_function in urls:
    soup = scrape_url(url)
    qna_pairs = extract_function(soup)
    dataset.extend(qna_pairs)

# Save updated dataset
with open(json_file_path, 'w') as f:
    json.dump(dataset, f, indent=4)

print(f"Dataset updated and saved to {json_file_path}")