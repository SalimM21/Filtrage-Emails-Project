# app.py
import streamlit as st
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt
from wordcloud import WordCloud # Assurez-vous d'avoir installé ces bibliothèques
import pandas as pd

# --- Configuration de la page ---
st.set_page_config(
    page_title="Détecteur de Spam",
    page_icon="✉️",
    layout="wide"
)

st.title("✉️ Détecteur de Spam - Application Interactive")
st.markdown("### Testez notre modèle en temps réel")

# --- Fonctions de Prétraitement (doivent correspondre exactement à l'entraînement) ---
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """Prétraite le texte pour le rendre compatible avec le modèle."""
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word not in stop_words]
    stemmed_tokens = [ps.stem(word) for word in filtered_tokens]
    return " ".join(stemmed_tokens)

# --- Chargement du Modèle et du Vectorizer ---
@st.cache_resource
def load_model_and_vectorizer():
    """Charge le modèle et le vectorizer depuis le disque."""
    try:
        model = joblib.load('models/best_spam_detector.pkl')
        vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
        return model, vectorizer
    except FileNotFoundError:
        st.error("Fichiers de modèle ou de vectorizer introuvables. Assurez-vous d'avoir entraîné le modèle et de l'avoir sauvegardé.")
        st.stop()

model, vectorizer = load_model_and_vectorizer()


# --- Section de Prédiction ---
st.header("Analyse d'un email")
user_input = st.text_area("Entrez le texte de l'email à analyser ici :", height=200)

if st.button("Analyser", use_container_width=True):
    if user_input:
        # Prétraiter et vectoriser l'entrée de l'utilisateur
        preprocessed_input = preprocess_text(user_input)
        vectorized_input = vectorizer.transform([preprocessed_input])

        # Faire la prédiction
        prediction = model.predict(vectorized_input)
        prediction_proba = model.predict_proba(vectorized_input)

        # Afficher le résultat
        st.subheader("Résultat de l'analyse :")
        
        # Pour récupérer le label 'ham' ou 'spam'
        # Le LabelEncoder est implicitement 0=ham, 1=spam
        prediction_label = "SPAM" if prediction[0] == 1 else "HAM"
        
        # Style de l'affichage
        if prediction_label == "SPAM":
            st.markdown(f"<div style='background-color:#FFD2D2; padding:10px; border-radius:5px; text-align:center;'><h2>🚨 {prediction_label}</h2></div>", unsafe_allow_html=True)
            st.warning(f"Probabilité de spam : **{prediction_proba[0][1]:.2f}**")
        else:
            st.markdown(f"<div style='background-color:#D4FFD4; padding:10px; border-radius:5px; text-align:center;'><h2>✅ {prediction_label}</h2></div>", unsafe_allow_html=True)
            st.success(f"Probabilité de ham : **{prediction_proba[0][0]:.2f}**")
    else:
        st.warning("Veuillez entrer un texte pour l'analyse.")


# --- Section EDA (Facultatif - à compléter avec vos données) ---
st.sidebar.title("Visualisations (EDA)")
st.sidebar.markdown("---")

st.sidebar.subheader("Nuage de mots")
st.sidebar.markdown("Pour afficher un nuage de mots du spam et du ham, décommentez la section ci-dessous et assurez-vous que votre DataFrame `df` est disponible.")
# Pour cette section, vous auriez besoin du DataFrame original chargé
if st.sidebar.button("Générer les nuages de mots"):
    try:
        df = pd.read_csv('DataSet_Emails.csv')
        ham_text = ' '.join(df[df['label_text'] == 'ham']['text'])
        spam_text = ' '.join(df[df['label_text'] == 'spam']['text'])

        st.write("### Nuage de Mots - Ham")
        wordcloud_ham = WordCloud(width=800, height=400, background_color='white').generate(ham_text)
        fig, ax = plt.subplots()
        ax.imshow(wordcloud_ham, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)

        st.write("### Nuage de Mots - Spam")
        wordcloud_spam = WordCloud(width=800, height=400, background_color='white').generate(spam_text)
        fig, ax = plt.subplots()
        ax.imshow(wordcloud_spam, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Erreur lors de la génération du nuage de mots : {e}")