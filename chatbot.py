import streamlit as st
import json
from transformers import RagTokenizer, RagTokenForGeneration
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the fine-tuned model and tokenizer
tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-base")
model = RagTokenForGeneration.from_pretrained("facebook/rag-token-base")

# Path to the JSON file
json_file_path = 'train_dataset.json'

# Load the dataset
def load_dataset(file_path):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return []

# Save the dataset
def save_dataset(dataset, file_path):
    with open(file_path, 'w') as f:
        json.dump(dataset, f, indent=4)

# Initialize the dataset
dataset = load_dataset(json_file_path)

st.title("Kimisays")

user_input = st.text_input("Enter your question:")

if user_input:
    # Initialize found flag
    found = False

    # Search for the matching question in the dataset
    response = "The answer to this question is not in my domain of knowledge."
    
    # Check for exact match
    for item in dataset:
        if user_input.lower() == item["question"].lower():
            response = item["context"]
            found = True
            break

    # If no exact match, use TF-IDF to find the most similar question
    if not found:
        questions = [item["question"] for item in dataset]
        if questions:
            vectorizer = TfidfVectorizer().fit_transform(questions + [user_input])
            vectors = vectorizer.toarray()
            cosine_similarities = cosine_similarity(vectors[-1].reshape(1, -1), vectors[:-1])
            most_similar_index = cosine_similarities.argsort()[0][-1]
            similarity_score = cosine_similarities[0][most_similar_index]
            
            # Set a threshold for similarity
            if similarity_score > 0.5:  # Adjust the threshold as needed
                response = dataset[most_similar_index]["context"]
                found = True

    # If still not found, generate a response using the model
    # If still not found, generate a response using the model
# If still not found, generate a response using the model
    if not found:
        inputs = tokenizer([user_input], return_tensors="pt")
        print("Inputs after tokenization:", inputs)  # Debug print
        
        if "input_ids" not in inputs or inputs["input_ids"].shape[1] == 0:
            st.write("Chatbot response:", response)
        else:
            # Remove None values from inputs
            inputs = {k: v for k, v in inputs.items() if v is not None}
            print("Filtered inputs:", inputs)  # Debug print
            
            try:
                generated_ids = model.generate(**inputs)
                response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            except AttributeError as e:
                print(f"Error during generation: {e}")  # Debug print
                response = "I'm sorry, I couldn't generate a response at this time."
            
            # Update the dataset with the new question and response (context)
            dataset.append({"question": user_input, "context": response})
            save_dataset(dataset, json_file_path)

    # Display the response
    st.write("Chatbot response:", response)