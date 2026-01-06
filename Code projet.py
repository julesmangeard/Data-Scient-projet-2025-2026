#Projet Data science pour le 10/01/2026


#Membres du groupe : Jules Mangeard ; Margot Schmutz ; Pierre Launay ; Eliot Nehme

#Projet : Isolation-based Anomaly Detection
#Jeu de données : fretelematic

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

data = pd.read_excel("chemin_vers_donnees") #Les données sont disponibles dans le GitHub soumis sous le nom de "data" et en format xlsx


data_fixed = data.copy()

##Préparation de nos données

#Comme prévu on enlève certaines colonnes (explication dans le PDF soumis) :

data = data.drop(columns=["Policy_ID", "Insured_Gender","Claim"])


#On applique le one hot encoding pour les variables non numériques, on va coder avec la variable "Low"

col_one_hot = ["Acceleration", "Brake", "Corner"]

data = pd.get_dummies(
    data,
    columns=col_one_hot,
    drop_first=True
).astype(int)


##Maintenant nos données sont prêtes pour appliquer l'algo d'isolation


from sklearn.ensemble import IsolationForest

iso = IsolationForest(
    n_estimators=200,      #Nombre d'arbres (explication du 200 dans le Cf PDF)
    contamination=0.05,   #Proportion attendue d'anomalies (explication du 0.05 dans le Cf PDF)
    random_state=42
)

X = data

iso.fit(X)


##Résultat de notre algorithme
scores = iso.decision_function(X)

labels = iso.predict(X)

data["Anomaly_score"] = scores
data["Anomaly_label"] = labels


data["Anomaly"] = (data["Anomaly_label"] == -1).astype(int)

data = data.drop(columns = ["Anomaly_label"])


#On visualise les 15 anomalies les plus prononcées :
data.sort_values("Anomaly_score").head(15)

##Anomalie par exemple ligne 821 : data.iloc[821,]


#On va visualiser la distribution du comptage des anomalies
#On arrondi au centième
scores_rounded = data["Anomaly_score"].round(2)
min_score = scores_rounded.min()
max_score = scores_rounded.max()

#Pas au centième
bins = np.arange(min_score, max_score + 0.01, 0.01)

plt.figure()
plt.hist(
    scores_rounded,
    bins=bins,
    alpha=0.8,
    label="Distribution des scores d'anomalie"
)

plt.axvline(
    x=0,
    linestyle="--",
    color="red",
    linewidth=2,
    label="Démarcation entre normal et anomalie"
)

plt.xlabel("Score anomalie")
plt.ylabel("Fréquence")
plt.title("Distribution des scores d'anomalie")
plt.legend()
plt.show()

##On analyse les anomalies

df_anomalie = data[data["Anomaly"] == 1]

df_normal = data[data["Anomaly"] == 0]

#Est-ce que l'on trouve une trend moyenne des anomalies ?

moyennes_anomalie = df_anomalie.mean()
moyennes = df_normal.mean()


#Dataframe résumant nos résultats moyens
df_stats = pd.DataFrame([moyennes_anomalie,moyennes], index=['Moyenne anomalie',"Moyenne valeurs normales"])


print(df_stats.iloc[0,])
print(df_stats.iloc[1,])

##Corrélation entre anomalie et sinistralité

liste_acc = data_fixed['Claim']


data = pd.concat([data,liste_acc], axis = 1)

data["Claim"] = (data["Claim"] == "yes").astype(int)

corr = data['Claim'].corr(data['Anomaly'])

print(corr)


