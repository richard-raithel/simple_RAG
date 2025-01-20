import os
import numpy as np
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv
import gradio as gr


load_dotenv()

# Initialize MongoDB client and database
client = MongoClient("mongodb://localhost:27017/")
db = client["rag_test_db"]
collection = db["documents"]

# Load the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# OpenAI API key setup
client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)


def encode_text(text):
    return model.encode([text])[0]


def retrieve_documents(query, top_k=5):
    query_embedding = encode_text(query)
    documents = collection.find()
    scores = []
    for doc in documents:
        doc_embedding = np.array(doc['embedding'])
        score = np.dot(query_embedding, doc_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding))
        scores.append((score, doc['text'], doc['document_id']))
    scores.sort(key=lambda x: x[0], reverse=True)
    top_docs = [(doc, doc_id) for score, doc, doc_id in scores[:top_k]]
    return top_docs


def generate_prompt(query, retrieved_docs):
    prompt = f"Answer the following query using only the information obtained from the retrieved documents. Reference the document ID where the information was found.\n\nQuery: {query}\n\nRetrieved Documents:\n"
    for i, (doc, doc_id) in enumerate(retrieved_docs):
        prompt += f"Document {i + 1} (ID: {doc_id}):\n{doc}\n\n"
    prompt += "Answer:"
    return prompt


def get_openai_response(prompt):
    completion = client.chat.completions.create(
        model="gpt-4o-2024-05-13",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return completion.choices[0].message.content


def main():
    query = input("Enter your query: ")
    retrieved_docs = retrieve_documents(query)
    if not retrieved_docs:
        print("No relevant documents found.")
        return
    else:
        print("Retrieved Documents:\n")
        for i, (doc, doc_id) in enumerate(retrieved_docs):
            print(f"Document {i + 1} (ID: {doc_id})\n")

    prompt = generate_prompt(query, retrieved_docs)
    response = get_openai_response(prompt)

    print("Generated Response:\n")
    print(response)


def query_system(message, history):
    query = message
    retrieved_docs = retrieve_documents(query)
    if not retrieved_docs:
        return history + [(query, "No relevant documents found.")]
    prompt = generate_prompt(query, retrieved_docs)
    response = get_openai_response(prompt)
    return response


# Create Gradio chatbot interface
iface = gr.ChatInterface(
    fn=query_system,
    title="Document Retrieval and Q&A System",
    description="Enter a query to search the document database and get a response based on the retrieved documents."
)

if __name__ == "__main__":
    iface.launch()
