# import pandas as pd
# import numpy as np
# import nltk
# from nltk.stem.snowball import SnowballStemmer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import streamlit as st
# from PIL import Image

# # Load the dataset
# data = pd.read_csv('C:/Users/himan/search engine recommendation/amazon_product.csv')

# # Remove unnecessary columns
# data = data.drop('id', axis=1)

# # Define tokenizer and stemmer
# stemmer = SnowballStemmer('english')
# def tokenize_and_stem(text):
#     tokens = nltk.word_tokenize(text.lower())
#     stems = [stemmer.stem(t) for t in tokens]
#     return stems

# # Create stemmed tokens column
# data['stemmed_tokens'] = data.apply(lambda row: tokenize_and_stem(row['Title'] + ' ' + row['Description']), axis=1)

# # Define TF-IDF vectorizer and cosine similarity function
# tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenize_and_stem)
# def cosine_sim(text1, text2):
#     # tfidf_matrix = tfidf_vectorizer.fit_transform([text1, text2])
#     text1_concatenated = ' '.join(text1)
#     text2_concatenated = ' '.join(text2)
#     tfidf_matrix = tfidf_vectorizer.fit_transform([text1_concatenated, text2_concatenated])
#     return cosine_similarity(tfidf_matrix)[0][1]

# # Define search function
# def search_products(query):
#     query_stemmed = tokenize_and_stem(query)
#     data['similarity'] = data['stemmed_tokens'].apply(lambda x: cosine_sim(query_stemmed, x))
#     results = data.sort_values(by=['similarity'], ascending=False).head(10)[['Title', 'Description', 'Category']]
#     return results

# # web app
# img = Image.open('img.png')
# st.image(img,width=600)
# st.title("Amazon Product Recommendation System on Dataset")
# query = st.text_input("Enter Product Name")
# sumbit = st.button('Search')
# if sumbit:
#     res = search_products(query)
#     st.write(res)
import pandas as pd
import numpy as np
import nltk
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from PIL import Image
import time
from collections import Counter
import matplotlib.pyplot as plt

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Configure the page
st.set_page_config(
    page_title="Amazon Product Recommender",
    page_icon="🛍️",
    layout="wide"
)

# Custom CSS to improve the UI
st.markdown("""
    <style>
    .stTitle {
        color: #FF9900;
        font-family: 'Arial Black', sans-serif;
    }
    .stTextInput {
        border-radius: 10px;
    }
    .stButton > button {
        background-color: #FF9900;
        color: white;
        border-radius: 20px;
        padding: 10px 25px;
        font-weight: bold;
    }
    .product-card {
        background-color: #23272f;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        color: #FF9900;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess the dataset with caching"""
    try:
        data = pd.read_csv('amazon_product.csv')
        data = data.drop('id', axis=1)
        return data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

@st.cache_resource
def initialize_nlp():
    """Initialize NLP components with caching"""
    stemmer = SnowballStemmer('english')
    vectorizer = TfidfVectorizer(tokenizer=tokenize_and_stem)
    return stemmer, vectorizer

def tokenize_and_stem(text):
    """Tokenize and stem the text"""
    if pd.isna(text):
        return []
    tokens = nltk.word_tokenize(str(text).lower())
    stems = [stemmer.stem(t) for t in tokens]
    return stems

def cosine_sim(text1, text2):
    """Calculate cosine similarity between two texts"""
    try:
        text1_concatenated = ' '.join(text1)
        text2_concatenated = ' '.join(text2)
        tfidf_matrix = vectorizer.fit_transform([text1_concatenated, text2_concatenated])
        return cosine_similarity(tfidf_matrix)[0][1]
    except Exception as e:
        st.error(f"Error calculating similarity: {str(e)}")
        return 0

def search_products(query, data):
    """Search for products based on query"""
    with st.spinner('Searching for products...'):
        query_stemmed = tokenize_and_stem(query)
        data['similarity'] = data['stemmed_tokens'].apply(lambda x: cosine_sim(query_stemmed, x))
        results = data.sort_values(by=['similarity'], ascending=False).head(10)
        return results[['Title', 'Description', 'Category', 'similarity']]

def main():
    # Load data and initialize components
    data = load_data()
    if data is None:
        return

    # Process data
    data['stemmed_tokens'] = data.apply(
        lambda row: tokenize_and_stem(str(row['Title']) + ' ' + str(row['Description'])), 
        axis=1
    )

    # Header
    col1, col2 = st.columns([1, 2])
    with col1:
        img = Image.open('img.png')
        st.image(img, width=300)
    with col2:
        st.title("Amazon Product Recommendation System")
        st.markdown("### Find the perfect products based on your interests! 🎯")

    # Search interface
    query = st.text_input("🔍 What are you looking for?", 
                         placeholder="Enter product name, description, or keywords...")
    
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        search_button = st.button('🔍 Search Products')

    if search_button and query:
        results = search_products(query, data)
        
        if not results.empty:
            st.markdown("### 🎯 Top Recommendations")
            # Show only product names as realistic tags
            product_names = results['Title'].tolist()
            st.markdown("#### Related Product Names:")
            st.markdown(
                " ".join([
                    f'<span style="background-color:#FF9900;color:white;padding:8px 16px;border-radius:20px;margin:4px;display:inline-block;font-weight:bold;">{name}</span>'
                    for name in product_names
                ]), unsafe_allow_html=True
            )
            st.markdown("---")
            # Visualize product name frequency (if duplicates) with a matplotlib bar chart
            name_freq = Counter(product_names)
            if name_freq:
                names, counts = zip(*name_freq.items())
            else:
                names, counts = [], []
            if names and counts:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.bar(names, counts, color='#FF9900')
                ax.set_title('Related Product Name Frequency (Matplotlib)')
                ax.set_xlabel('Product Name')
                ax.set_ylabel('Frequency')
                plt.xticks(rotation=45, ha='right')
                st.pyplot(fig)
            # Show product cards for details
            for _, row in results.iterrows():
                similarity_percentage = int(row['similarity'] * 100)
                highlight = '#28a745' if similarity_percentage > 70 else '#FF9900'
                with st.container():
                    st.markdown(f"""
                    <div class="product-card">
                        <h3 style=\"color: {highlight};\">{row['Title']}</h3>
                        <p><strong>Category:</strong> {row['Category']}</p>
                        <p><strong>Match Score:</strong> {similarity_percentage}%</p>
                        <p><em>{row['Description'][:200]}...</em></p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No products found matching your search. Try different keywords!")

if __name__ == "__main__":
    # Initialize NLP components
    stemmer, vectorizer = initialize_nlp()
    main()
