import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from model import recommend
from data import items, item_names

st.title("🎯 Smart Recommendation System (Softmax)")

st.write("Type what you like (example: action movie, music, fitness, technology)")

# 👉 User input (REAL WORLD STYLE)
user_input = st.text_input("Enter your interests:")

if user_input:

    # Convert text → vectors
    vectorizer = TfidfVectorizer()
    item_vectors = vectorizer.fit_transform(items).toarray()
    user_vector = vectorizer.transform([user_input]).toarray()[0]

    scores, probs, recommendation = recommend(user_vector, item_vectors, item_names)

    st.subheader("📊 Scores")
    for name, score in zip(item_names, scores):
        st.write(f"{name}: {score:.3f}")

    st.subheader("📈 Probabilities (Softmax)")
    for name, prob in zip(item_names, probs):
        st.write(f"{name}: {prob:.3f}")

    st.bar_chart(probs)

    st.success(f"✅ Recommended for you: {recommendation}")