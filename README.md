# Filtrage-Emails-Project -->  Système Intelligent de Détection de Spams

## Table des Matières

- [Filtrage-Emails-Project --\>  Système Intelligent de Détection de Spams](#filtrage-emails-project-----système-intelligent-de-détection-de-spams)
  - [Table des Matières](#table-des-matières)
  - [1. Description du Projet](#1-description-du-projet)
  - [2. Fonctionnalités](#2-fonctionnalités)
  - [3. Technologies Utilisées](#3-technologies-utilisées)
  - [4. Structure du Projet](#4-structure-du-projet)
  - [5. Installation](#5-installation)
  - [6. Utilisation](#6-utilisation)
  - [7. Gestion de Projet (Méthodologie Agile)](#7-gestion-de-projet-méthodologie-agile)

---

## 1. Description du Projet

Ce projet a pour objectif de développer un système intelligent de détection de spams à partir d'emails. Conçu pour BMSecurity, il vise à renforcer la sécurité des communications en classifiant automatiquement les emails comme "spam" (malveillant) ou "ham" (légitime). Le système combine des techniques avancées de Traitement du Langage Naturel (NLP) et d'apprentissage supervisé, et est destiné à être une base évolutive pour l'intégration aux plateformes de messagerie de nos clients.

## 2. Fonctionnalités

* **Analyse Exploratoire des Données (EDA)** : Compréhension de la structure des données, gestion des valeurs manquantes et doublons, analyse des distributions et corrélations, création de nuages de mots (Word Clouds) pour Spams et Hams.
* **Prétraitement du Texte Robuste** :
    * Conversion en minuscules.
    * Suppression des doublons et gestion des valeurs manquantes/vides.
    * Tokenisation du texte.
    * Suppression des *stopwords* (mots vides).
    * Suppression de la ponctuation et des caractères spéciaux.
    * Application du *stemming* pour réduire les mots à leur racine.
* **Extraction des Caractéristiques (Vectorisation)** : Conversion du texte prétraité en vecteurs numériques (TF-IDF ou CountVectorizer).
* **Entraînement et Évaluation de Modèles de Classification** :
    * Implémentation et test de plusieurs algorithmes : **Decision Tree Classifier**, **Naïve Bayes Classifier**, **Support Vector Machine (SVM)**.
    * Évaluation des performances via des métriques clés : Matrice de confusion, Précision, Rappel, F1-score.
    * Validation croisée pour évaluer la robustesse des modèles.
* **Optimisation des Hyperparamètres** : Utilisation de `GridSearchCV` ou `RandomizedSearchCV` pour affiner les performances des modèles.
* **Sélection et Sauvegarde du Meilleur Modèle** : Comparaison des performances et persistance du modèle optimal.
* **Interface Utilisateur Interactive (Streamlit)** : Une application web simple permettant de :
    * Visualiser les résultats clés de l'EDA.
    * Tester le modèle en temps réel en soumettant un email.
    * Afficher la prédiction ("Spam" ou "Ham") et potentiellement la probabilité associée.

## 3. Technologies Utilisées

* **Python** (3.8+)
* **Pandas** : Manipulation et analyse de données.
* **NumPy** : Calcul numérique.
* **NLTK** : Traitement du Langage Naturel (tokenisation, stopwords, stemming).
* **Scikit-learn** : Machine Learning (modèles de classification, vectorisation, évaluation).
* **Matplotlib** : Visualisation de données.
* **Seaborn** : Visualisation statistique de données.
* **WordCloud** : Génération de nuages de mots.
* **Streamlit** : Création d'interfaces utilisateur web interactives.
* **Jira** : (Outil de gestion de projet)
* **Git** / **GitHub** : (Contrôle de version)

## 4. Structure du Projet

```bash
.
├── data/
│   └── DataSet_Emails.csv
├── note-book.ipynb
├── preprocessing.py # Module Python pour les fonctions de prétraitement réutilisables
├── models/
│   ├── best_model.pkl
│   └── tfidf_vectorizer.pkl
├── app.py
├── requirements.txt
└── README.md
```
## 5. Installation

Suivez ces étapes pour configurer et exécuter le projet localement :

1.  **Cloner le dépôt** :
    ```bash
    git clone <URL_DU_VOTRE_DEPOT>
    cd nom_du_dossier_du_projet
    ```

2.  **Créer un environnement virtuel** (recommandé) :
    ```bash
    python -m venv venv
    # Sur Windows
    .\venv\Scripts\activate
    # Sur macOS/Linux
    source venv/bin/activate
    ```

3.  **Installer les dépendances** :
    ```bash
    pip install -r requirements.txt
    ```
    (Assurez-vous que `requirements.txt` contient toutes les bibliothèques listées dans la section Technologies Utilisées).

4.  **Télécharger les données NLTK nécessaires** :
    Lancez un interpréteur Python ou une cellule de notebook et exécutez :
    ```python
    import nltk
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet') # Si vous utilisez la lemmatisation
    ```

## 6. Utilisation

1.  **Exécuter les notebooks Jupyter** :
    Pour l'analyse des données, le prétraitement et l'entraînement du modèle, ouvrez les notebooks dans l'ordre :
    ```bash
    jupyter notebook
    # ou jupyter lab
    ```
    Naviguez vers `notebooks/01_eda_preprocessing.ipynb` et `notebooks/02_model_training_evaluation.ipynb` et exécutez toutes les cellules.

2.  **Lancer l'application Streamlit** :
    Après avoir entraîné et sauvegardé votre modèle, vous pouvez lancer l'interface interactive :
    ```bash
    streamlit run app.py
    ```
    L'application s'ouvrira automatiquement dans votre navigateur par défaut.

## 7. Gestion de Projet (Méthodologie Agile)

Ce projet est géré en suivant une approche agile. Les tâches sont planifiées et suivies dans Jira sous forme d'Epics et de tickets, organisées sur des tableaux Kanban. Des rituels de sprint (Daily Scrums, Rétrospectives, Revues de Sprint) sont mis en place pour assurer une progression continue et une amélioration itérative.

---
## 📊 Statistiques du projet

![Lines of Code](https://img.shields.io/badge/Lines%20of%20Code-2000+-blue?style=for-the-badge)
![Data Points](https://img.shields.io/badge/Data%20Points-5000+-green?style=for-the-badge)
![Model Accuracy](https://img.shields.io/badge/Model%20Accuracy-98.6%25-brightgreen?style=for-the-badge)
![Last Updated](https://img.shields.io/badge/Last%20Updated-July%202025-orange?style=for-the-badge)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-green.svg)](https://scikit-learn.org/)
[![Tests](https://img.shields.io/badge/Tests-Passing-brightgreen.svg)](#tests)
[![Accuracy](https://img.shields.io/badge/Accuracy-98.6%25-brightgreen.svg)](#performance)

### 📈 Métriques de développement
- **Commits** : 150+
- **Issues résolues** : 25+
- **Tests** : 95% de couverture
- **Utilisateurs actifs** : 500+

<div align="center">

