import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from datasets import load_dataset
import os

def ecommerce_knowledge_base():
    # Create a vector database from Bitext e-commerce dataset
    print ("Loading Bitext e-commerce dataset...")

    # Load the dataset
    dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")
    print(f"Dataset loadedL {len(dataset['train'])} examples")

    # Prepare knowledge base
    knowledge_base = []
    for example in dataset['train']:
        # Clean the responses
        response = example['response']
        response = response.replace("{{Order Number}}", "your order")
        response = response.replace("{{Online Company Portal Info}}", "our website")
        response = response.replace("{{Online Order Interaction}}", "order history")
        response = response.replace("{{Customer Support Hours}}", "business hours")
        response = response.replace("{{Customer Support Phone Number}}", "our support line")
        response = response.replace("{{Website URL}}", "our website")\
        
        knowledge_base.append({
           'question': example['instruction'],
           'answer': response,
           'intent': example['intent'],
           'category': example['category']
        })
    print (f"Knowledge base created with {len(knowledge_base)} entries")

    # Load sentence transformer for embeddings
    print("Loading sentence transformer model...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Create embeddings for all question
    print("Creating embeddings...")
    questions = [item['question'] for item in knowledge_base]
    embeddings = model.encode(questions, show_progress_bar=True)

    # Create FAISS index
    print("Creating FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension) # Inner Product for similarity

    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)
    index.add(embeddings.astype('float32'))

    # Save everything
    print ("Saving knowledge base and index...")

    # Getting script file path
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Save knowledge base
    kb = os.path.join(script_dir, 'knowledge_base.pkl') 
    with open (kb, 'wb') as f:
        pickle.dump(knowledge_base, f)

    # Save FAISS index
    ff = os.path.join(script_dir,'ecommerce_index.faiss')
    faiss.write_index(index, ff)

    # Save model name for later loading
    mn = os.path.join(script_dir,'model_name.txt')
    with open(mn, 'w') as f:
        f.write('sentence-transformers/all-MiniLM-L6-v2')

    print("Knowledge base successfully created!")

if __name__ == "__main__":
    ecommerce_knowledge_base()






