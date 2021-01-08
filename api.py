#! usr/bin/python3.7
# -*- coding: ISO-8859-1 -*-

from flask import Flask, jsonify, request, make_response
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from utils import preprocess_dataset, scale_dataset
from tqdm import tqdm
import time

# barre de chargement
pbar = tqdm(
    range(0, 6), desc="Import des données")

# 1] Import des données ------------------------

data_set = pd.read_csv("data/OnlineNewsPopularity.csv", sep=";")
data_set.shape

pbar.update(1)
pbar.set_description(
    "Préparation des données")

# 2] Nettoyage des données +  Discrétisation de la colonne cible ------------------------

data_set = preprocess_dataset(data_set)

pbar.update(1)

# 3] Standardisation des données ------------------------

data_set_strd = scale_dataset(data_set)

pbar.update(1)

# 4] Split des données ------------------------

# Split du dataset en X (features) et y (target)
X = data_set_strd.drop(columns=['popularity', 'shares', 'url'])
y = data_set_strd['popularity'].values.tolist()

# Split du data_set en train et test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

pbar.update(1)

# 5] Mise en place du modèle ------------------------

pbar.set_description(
    "Mise en place du modèle de prédiction")

rf_opt = RandomForestClassifier(n_estimators=4980, n_jobs=-1, random_state=42)
rf_opt.fit(X_train, y_train)

pbar.update(1)

# 6] Préparation des données pour l'api ------------------------

pbar.set_description(
    "Préparation de l'api")

# création de la liste d'article
artcl_titles = data_set['url'].apply(lambda x: x.split('/')[-2]).tolist()

# création de notre base de donnée de prédiction
X_api = data_set_strd.drop(columns=['popularity', 'shares', 'url'])
Y_api = data_set_strd['popularity'].tolist()

# on récupère notre modèle
model = rf_opt

pbar.update(1)

# 7] Mise en place de l'api ------------------------

pbar.set_description(
    "Lancement de l'api")

# création de l'api
app = Flask(__name__)


#### End-point permettant de vérifier que l'api est bien lancé ####
@app.route('/')
def index():
    return '<h1>API Launched</h1>'


#### End point permettant de récuppérer la liste des article disponible à la prédiction ####
@app.route('/articles', methods=['GET'])
def get_articles():

    return jsonify({"articles": artcl_titles}), 200


#### End point permettant de récuppérer la liste des article disponible à la prédiction ####
@app.route('/popularity', methods=['POST'])
def get_popularity():

    # Récupérer la requette en json ------------------------------

    req = request.get_json()

    if "article" in list(req.keys()):
        if(type(req["article"]) == list):
            titles = req["article"]
        else:
            titles = [req["article"]]
    else:
        return jsonify({'success': False, 'error': "aucun article n'a été selectionné par l\'api"}), 500

    # Étape 1: Récupération des articles ------------------------------
    artcls_index = []

    try:
        for title in titles:
            artcls_index.append(artcl_titles.index(title))
    except ValueError:
        return jsonify({'success': False, 'error': "Un ou plusieurs des articles selectionnés n'existe pas."}), 500
    artcls = X_api.iloc[artcls_index]

    # Étape 2: Prédiction ------------------------------

    result = model.predict(artcls)

    return jsonify({"result": [{"article": title, "popularity": result[index]} for index, title in enumerate(titles)]}), 200


pbar.update(1)

# Démarrer l'API
app.run()
