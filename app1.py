import dash
from dash import html, dcc, Output, Input, State
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, classification_report
import dash_bootstrap_components as dbc
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize app first
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# Load and prepare data
try:
    df = pd.read_csv("DataSet_Emails.csv")
    # Ensure required columns exist
    if 'text' not in df.columns:
        df['text'] = ""
    if 'label' not in df.columns:
        df['label'] = 0
        
    df["text"] = df["text"].fillna("").astype(str)
    df["length"] = df["text"].str.len()
except Exception as e:
    print(f"Error loading data: {e}")
    # Create sample data if loading fails
    df = pd.DataFrame({
        "text": ["Sample ham email", "Sample spam email", "Another normal email"],
        "label": [0, 1, 0],
        "length": [15, 16, 18]
    })

# Load or create models
try:
    model = joblib.load("models/best_spam_detector.pkl")
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    print("Successfully loaded pre-trained models")
except Exception as e:
    print(f"Error loading models: {e}. Creating and training new models...")
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df["text"])
    y = df["label"]
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    print("Created and trained new models")

# Define Predictor class (fixed typo in class name)
class Predictor:
    def __init__(self, model, vectorizer):
        self.model = model
        self.vectorizer = vectorizer
    
    def predict(self, text):
        try:
            text = str(text).strip()
            if not text:
                return 0, 0.0  # Default to HAM if empty
            
            X = self.vectorizer.transform([text])
            label = self.model.predict(X)[0]
            proba = max(self.model.predict_proba(X)[0])  # Get max probability
            return label, proba
        except Exception as e:
            print(f"Prediction error: {e}")
            return 0, 0.0  # Fallback to HAM on error

# Initialize predictor
predictor = Predictor(model, vectorizer)

# App layout
app.layout = dbc.Container([
    html.H1("\U0001F4E7 Spam Classifier", className="my-3 text-center"),
    html.P("Ce modèle prédit si un email est SPAM ou HAM (non-spam).", className="mb-4 text-center"),
    
    dbc.Card([
        dbc.CardHeader("\U0001F50D Tester un email"),
        dbc.CardBody([
            dcc.Textarea(
                id='email-input', 
                placeholder="Entrer le contenu de l'email", 
                style={"width": "100%", "height": 150}
            ),
            html.Br(),
            dbc.Button("Prédire", id='predict-button', color="primary", className="mt-2"),
            html.Div(id='prediction-output', className="mt-3")
        ])
    ], className="mb-4"),

    dbc.Card([
        dbc.CardHeader("\U0001F4CA Analyse exploratoire (EDA)"),
        dbc.CardBody([
            html.H5("Distribution des classes"),
            dcc.Graph(
                figure=px.histogram(
                    df, 
                    x="label", 
                    title="Distribution des classes",
                    labels={"label": "Type d'email", "count": "Nombre"},
                    category_orders={"label": ["0", "1"]}
                ).update_layout(xaxis_title="0 = HAM, 1 = SPAM")
            ),

            html.H5("Longueur des messages"),
            dcc.Graph(
                figure=px.histogram(
                    df, 
                    x="length", 
                    color="label", 
                    nbins=50, 
                    title="Longueur des messages par classe",
                    labels={"length": "Longueur du message", "count": "Nombre"}
                )
            )
        ])
    ], className="mb-4"),

    dbc.Card([
        dbc.CardHeader("\U0001F4C9 Évaluation du modèle"),
        dbc.CardBody([
            html.H5("Matrice de confusion"),
            dcc.Graph(id='confusion-matrix'),

            html.H5("Rapport de classification"),
            dcc.Loading(dcc.Graph(id='classification-report'))
        ])
    ]),

    html.Hr(),
    html.P("\U0001F4E6 Projet IA · Modèle TF-IDF + Classifieur supervisé", className="text-center"),
    
    # Hidden div for debug output
    html.Div(id='debug-output', style={'display': 'none'})
])

# Callbacks
@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-button', 'n_clicks'),
    State('email-input', 'value'),
    prevent_initial_call=True
)
def predict_email(n_clicks, text):
    if not n_clicks or not text:
        raise dash.exceptions.PreventUpdate
    
    try:
        label, proba = predictor.predict(text)
        if label == 1:
            return dbc.Alert(
                f"✉️ SPAM détecté (confiance : {proba:.2%})", 
                color="danger",
                className="text-center"
            )
        else:
            return dbc.Alert(
                f"✓ HAM (non-spam) (confiance : {proba:.2%})", 
                color="success",
                className="text-center"
            )
    except Exception as e:
        return dbc.Alert(
            f"Erreur de prédiction: {str(e)}", 
            color="warning",
            className="text-center"
        )

@app.callback(
    Output('confusion-matrix', 'figure'),
    Output('classification-report', 'figure'),
    Input('predict-button', 'n_clicks')
)
def update_eval_section(n):
    try:
        y_true = df["label"]
        X_transformed = vectorizer.transform(df["text"])
        y_pred = model.predict(X_transformed)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        labels = ["HAM", "SPAM"]
        fig_cm = px.imshow(
            cm, 
            text_auto=True, 
            x=labels, 
            y=labels, 
            color_continuous_scale='Blues',
            labels=dict(x="Prédiction", y="Réel", color="Nombre"),
            title="Matrice de confusion"
        )

        # Classification report
        report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
        df_report = pd.DataFrame(report).transpose().reset_index().rename(columns={"index": "Classe"})
        fig_report = px.bar(
            df_report[df_report["Classe"].isin(labels)], 
            x="Classe", 
            y=["precision", "recall", "f1-score"],
            barmode='group',
            title="Métriques par classe",
            labels={"value": "Score", "variable": "Métrique"}
        )
        
        return fig_cm, fig_report
    
    except Exception as e:
        print(f"Error in evaluation: {e}")
        # Return empty figures if error occurs
        return px.scatter(title="Error generating confusion matrix"), px.scatter(title="Error generating classification report")

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=8051)