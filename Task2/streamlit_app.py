# streamlit_app.py

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud
from nlp_pipeline import preprocess_text, get_word_freq, get_pos_tags, get_ngrams, tokens_to_dataframe, get_top_pos

st.set_page_config(page_title="Text Analytics App", layout="wide")

st.title(" NLTK-Powered Text Analytics Web App")

uploaded_file = st.file_uploader("Upload a .txt file", type="txt")

if uploaded_file:
    raw_text = uploaded_file.read().decode("utf-8")
    st.subheader(" Raw Text Preview")
    st.text_area("Content", raw_text[:1000] + "...", height=200)

    st.subheader("🧹 Preprocessing...")
    tokens = preprocess_text(raw_text)
    st.write(f"Total Tokens after cleaning: {len(tokens)}")
    
    st.subheader(" Word Frequency")
    word_freq = get_word_freq(tokens)
    st.bar_chart(pd.DataFrame(word_freq, columns=["Word", "Frequency"]).set_index("Word"))

    st.subheader("Word Cloud")
    wordcloud = WordCloud(width=800, height=300, background_color='white').generate(" ".join(tokens))
    st.image(wordcloud.to_array(), use_column_width=True)

    st.subheader(" POS Tags Distribution")
    pos_tags = get_pos_tags(tokens)
    top_pos = get_top_pos(pos_tags)
    st.bar_chart(pd.DataFrame(top_pos, columns=["POS", "Count"]).set_index("POS"))

    st.subheader(" Top Bigrams")
    bigrams = get_ngrams(tokens, n=2)
    st.write(pd.DataFrame(bigrams, columns=["Bigram", "Count"]))
