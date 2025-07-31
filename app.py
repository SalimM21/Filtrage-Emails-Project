import streamlit as st
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd

# --- Configuration de la page Streamlit ---
st.set_page_config(
    page_title="Détecteur de Spam",
    page_icon="✉️",
    layout="wide" # Utilise toute la largeur de l'écran
)

# --- Titre principal de l'application ---
st.title("✉️ Application de Détection de Spam")

# --- Fonctions de Prétraitement (doivent être identiques à celles de l'entraînement) ---

# Télécharge les ressources NLTK une seule fois et les met en cache
@st.cache_resource
def load_nltk_resources():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

load_nltk_resources() # Appelle la fonction pour charger les ressources NLTK

# Initialise le PorterStemmer et les stopwords
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """
    Prétraite le texte en le convertissant en minuscules, supprimant les chiffres,
    la ponctuation, les stopwords et en appliquant le stemming.
    """
    if not isinstance(text, str): # Gère les entrées non-texte
        return ""
    
    text = text.lower() # Convertit en minuscules
    text = re.sub(r'\d+', '', text) # Supprime les chiffres
    
    # Supprime la ponctuation
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    
    tokens = word_tokenize(text) # Tokenisation
    filtered_tokens = [word for word in tokens if word not in stop_words] # Suppression des stopwords
    stemmed_tokens = [ps.stem(word) for word in filtered_tokens] # Stemming
    
    return " ".join(stemmed_tokens) # Rejoint les tokens en une chaîne

# --- Chargement du Modèle, du Vectorizer et des Données pour l'EDA ---

# Met en cache les assets pour éviter de les recharger à chaque interaction
@st.cache_resource
def load_assets():
    """
    Charge le modèle de détection de spam, le vectorizer TF-IDF et le DataFrame original.
    Gère les erreurs si les fichiers ne sont pas trouvés.
    """
    try:
        # Charge le modèle entraîné (le meilleur, choisi lors de l'optimisation)
        model = joblib.load('models/best_spam_detector.pkl')
        # Charge le vectorizer TF-IDF entraîné
        vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
        # Charge le DataFrame original pour l'EDA
        df_eda = pd.read_csv('DataSet_Emails.csv')
        # Nettoyage initial du DataFrame pour l'EDA (gestion des NaN et doublons)
        df_eda.dropna(subset=['text', 'label_text'], inplace=True)
        df_eda['text'] = df_eda['text'].astype(str)
        df_eda.drop_duplicates(inplace=True)
        return model, vectorizer, df_eda
    except FileNotFoundError as e:
        st.error(f"Erreur: Un fichier essentiel est introuvable. Assurez-vous d'avoir entraîné le modèle et que 'DataSet_Emails.csv' est dans le dossier racine. Détail: {e}")
        st.stop() # Arrête l'exécution de l'application
    except Exception as e:
        st.error(f"Une erreur inattendue est survenue lors du chargement des assets: {e}")
        st.stop()

# Charge les assets au démarrage de l'application
model, vectorizer, df_original = load_assets()

# --- Navigation dans la barre latérale (Sidebar) ---
st.sidebar.title("Navigation")
selected_page = st.sidebar.radio(
    "Aller à :",
    ["📊 Analyse Exploratoire des Données (EDA)", "🤖 Tester le Modèle"]
)

# --- Contenu principal basé sur la sélection de la barre latérale ---
if selected_page == "📊 Analyse Exploratoire des Données (EDA)":
    st.header("Analyse Exploratoire des Données")
    st.markdown("Explorez la distribution des emails et les mots les plus fréquents dans chaque catégorie.")

    # 1. Graphique de la distribution des labels (Spam vs. Ham)
    st.subheader("Distribution des Emails (Spam vs. Ham)")
    label_counts = df_original['label_text'].value_counts()
    
    # Crée une figure Matplotlib
    fig_dist, ax_dist = plt.subplots(figsize=(8, 5))
    ax_dist.bar(label_counts.index, label_counts.values, color=['#4CAF50', '#FF5733']) # Couleurs personnalisées
    ax_dist.set_title("Nombre d'emails par catégorie", fontsize=16)
    ax_dist.set_xlabel("Catégorie", fontsize=12)
    ax_dist.set_ylabel("Nombre d'emails", fontsize=12)
    ax_dist.tick_params(axis='x', labelsize=10)
    ax_dist.tick_params(axis='y', labelsize=10)
    ax_dist.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Affiche le graphique dans Streamlit
    st.pyplot(fig_dist)
    st.write(f"Total d'emails analysés : {len(df_original)}")
    st.write(f"Emails Ham (légitimes) : {label_counts.get('ham', 0)} ({label_counts.get('ham', 0)/len(df_original)*100:.2f}%)")
    st.write(f"Emails Spam : {label_counts.get('spam', 0)} ({label_counts.get('spam', 0)/len(df_original)*100:.2f}%)")


    # 2. Nuages de mots pour Ham et Spam
    st.subheader("Nuages de mots des Emails")
    st.markdown("Les nuages de mots visualisent la fréquence des termes dans les emails légitimes (Ham) et les spams.")
    
    col_ham_wc, col_spam_wc = st.columns(2) # Crée deux colonnes pour les nuages de mots

    with col_ham_wc:
        st.markdown("#### Mots les plus fréquents dans les emails HAM")
        # Concatène tout le texte des emails "ham"
        ham_text = ' '.join(df_original[df_original['label_text'] == 'ham']['text'])
        # Génère le nuage de mots
        wordcloud_ham = WordCloud(width=600, height=300, background_color='white', colormap='viridis').generate(ham_text)
        
        # Crée une figure Matplotlib pour le nuage de mots Ham
        fig_ham_wc, ax_ham_wc = plt.subplots(figsize=(10, 5))
        ax_ham_wc.imshow(wordcloud_ham, interpolation='bilinear')
        ax_ham_wc.axis("off") # Cache les axes
        st.pyplot(fig_ham_wc) # Affiche le graphique dans Streamlit

    with col_spam_wc:
        st.markdown("#### Mots les plus fréquents dans les emails SPAM")
        # Concatène tout le texte des emails "spam"
        spam_text = ' '.join(df_original[df_original['label_text'] == 'spam']['text'])
        # Génère le nuage de mots avec une palette de couleurs différente
        wordcloud_spam = WordCloud(width=600, height=300, background_color='black', colormap='Reds').generate(spam_text)
        
        # Crée une figure Matplotlib pour le nuage de mots Spam
        fig_spam_wc, ax_spam_wc = plt.subplots(figsize=(10, 5))
        ax_spam_wc.imshow(wordcloud_spam, interpolation='bilinear')
        ax_spam_wc.axis("off") # Cache les axes
        st.pyplot(fig_spam_wc) # Affiche le graphique dans Streamlit


elif selected_page == "🤖 Tester le Modèle":
    st.header("Tester le modèle en temps réel")
    st.markdown("Entrez un email ci-dessous pour voir s'il est classifié comme Spam ou Ham.")
    
    user_input = st.text_area("Texte de l'email :", height=250, placeholder="Ex: Congratulations! You've won a free prize. Click here to claim.")

    if st.button("Analyser l'email", use_container_width=True):
        if user_input:
            # Prétraiter l'entrée de l'utilisateur
            preprocessed_input = preprocess_text(user_input)
            # Vectoriser l'entrée avec le vectorizer entraîné
            vectorized_input = vectorizer.transform([preprocessed_input])

            # Faire la prédiction
            prediction = model.predict(vectorized_input)
            # Obtenir les probabilités de prédiction
            prediction_proba = model.predict_proba(vectorized_input)

            st.subheader("Résultat de l'analyse :")
            
            # Déterminer le label et la probabilité correspondante
            # Assumant que 0 est 'ham' et 1 est 'spam' d'après LabelEncoder
            if prediction[0] == 1: # Si la prédiction est 1 (spam)
                prediction_label = "SPAM"
                probability = prediction_proba[0][1] # Probabilité d'être spam
                st.markdown(f"<div style='background-color:#FFD2D2; padding:15px; border-radius:8px; text-align:center; border: 2px solid #FF0000;'><h3>🚨 {prediction_label}</h3></div>", unsafe_allow_html=True)
                st.warning(f"Probabilité d'être SPAM : **{probability:.2f}**")
            else: # Si la prédiction est 0 (ham)
                prediction_label = "HAM"
                probability = prediction_proba[0][0] # Probabilité d'être ham
                st.markdown(f"<div style='background-color:#D4FFD4; padding:15px; border-radius:8px; text-align:center; border: 2px solid #008000;'><h3>✅ {prediction_label}</h3></div>", unsafe_allow_html=True)
                st.success(f"Probabilité d'être HAM : **{probability:.2f}**")
        else:
            st.info("Veuillez entrer un texte d'email pour effectuer l'analyse.")

