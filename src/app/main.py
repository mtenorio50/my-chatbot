import streamlit as st
import pickle
import faiss
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import os
folder = "/utils/"


st.set_page_config(page_title="Support Chat", page_icon="💬")


@st.cache_resource
def load_sytem():
    try:
        folder = "src/utils/"
        required_files = ['knowledge_base.pkl',
                          'ecommerce_index.faiss', 'model_name.txt']

        # Check if all required files exist
        for file in required_files:
            file_path = os.path.join(folder, file)
            if not os.path.exists(file_path):
                st.error(f"Missing: {file_path}")
                return None, None, None

        # Load model name
        with open(os.path.join(folder, 'model_name.txt'), 'r') as f:
            model_name = f.read().strip()

        model = SentenceTransformer(model_name)

        # Load knowledge base
        with open(os.path.join(folder, 'knowledge_base.pkl'), 'rb') as f:
            knowledge_base = pickle.load(f)

        # Load FAISS index
        index = faiss.read_index(os.path.join(folder, 'ecommerce_index.faiss'))

        st.success(f"Loaded knowledge base with {len(knowledge_base)} entries")
        return model, knowledge_base, index

    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None, None, None


def get_answer(query, model, knowledge_base, index):
    try:
        query_embedding = model.encode([query])
        faiss.normalize_L2(query_embedding)

        scores, indices = index.search(query_embedding.astype('float32'), 3)
        best_idx = indices[0][0]
        best_score = scores[0][0]

        if best_score < 0.3:
            return get_fallback(query)

        best_match = knowledge_base[best_idx]

        return {
            'answer': best_match['answer'],
            'confidence': "High" if best_score > 0.7 else "Medium"
        }

    except Exception as e:
        return {'answer': f"Sorry, error occurred: {str(e)}", 'confidence': 'Low'}


def get_fallback(query):
    query_lower = query.lower()

    responses = {
        'track': "Track your order in 'My Account' > 'Order History'",
        'return': "We offer 30-day returns. Start in your account",
        'refund': "Refund proicess in 5-7 days business days",
        'cancel': "Cancel orders within 1 hour in your account",
        'shipping': "Standard: 3-5 days, Express: 1-2 days",
        'payment': "We accept cards, PayPal, Apple Pay, Google Pay"
    }

    for keyword, response in responses.items():
        if keyword in query_lower:
            return {'answer': response, 'confidence': 'Medium'}
    return {
        'answer': "I'm here to help! Ask about orders, shipping, returns, or payments.",
        'confidence': 'Low'
    }


def main():
    st.title("💬 Customer Support")
    model, knowledge_base, index = load_sytem()
    if not all([model, knowledge_base, index]):
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask me anything..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            response = get_answer(prompt, model, knowledge_base, index)
            st.write(response['answer'])

            st.session_state.messages.append({
                "role": "assistant",
                "content": response['answer']
            })


if __name__ == "__main__":
    main()
