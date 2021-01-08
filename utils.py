from sklearn.preprocessing import StandardScaler
import pandas as pd


def preprocess_dataset(data_set):
    # 1] Nettoyage des données ------------------------

    # suppression de la colonne time delta
    data_set.drop(columns=['timedelta'], inplace=True)

    # supprimer les 1181 lignes qui semble contenir des erreurs (article vide)
    data_set = data_set[data_set['n_tokens_content'] != 0]

    # réindexer le dataset
    data_set = data_set.reset_index(drop=True)

    # 2] Discrétisation de la colonne cible ------------------------
    quantils = data_set.shares.quantile([0.10, 0.35, 0.65, 0.90])

    # on transforme quantils en list pour simplifier la suite
    quantils = quantils.tolist()

    # valeur de notre target
    labels = ['non populaire', 'peu populaire',
              'neutre', 'populaire', 'très populaire']

    # remplacer les valeurs de shares par les niveaux de classification
    for i, quantil in enumerate(quantils):
        data_set.loc[data_set.shares >= quantil, 'popularity'] = labels[i+1]
        if (i == 0):
            data_set.loc[data_set.shares < quantil, 'popularity'] = labels[i]

    return data_set


def scale_dataset(data_set):

    # on récupère les colonnes weekdays
    publish_day_columns = data_set.columns.values[list(data_set.columns).index(
        'weekday_is_monday'):list(data_set.columns).index('weekday_is_sunday')+1]
    publish_day_columns = publish_day_columns.tolist()

    # on récupère les colonnes channel
    channel_columns = data_set.columns.values[list(data_set.columns).index(
        'data_channel_is_lifestyle'):list(data_set.columns).index('data_channel_is_world')+1]
    channel_columns = channel_columns.tolist()

    # on récupère les colonnes lda
    lda_columns = data_set.columns.values[list(data_set.columns).index(
        'LDA_00'):list(data_set.columns).index('LDA_04')+1]
    lda_columns = lda_columns.tolist()

    # liste des colonnes à ne pas standardiser
    columns = publish_day_columns + channel_columns + \
        lda_columns + ['popularity', 'shares', 'is_weekend', 'url']
    columns

    scaler = StandardScaler()
    feature_names = data_set.drop(columns=columns).columns

    # standardisation des colonnes numérique
    data_feature_strd = scaler.fit_transform(
        data_set.drop(columns=columns).values)

    # Création d'un nouveau data_frame
    data_set_strd = pd.DataFrame(
        data_feature_strd, columns=feature_names).join(data_set[columns])

    return data_set_strd
