#Projet Data science pour le 10/01/2026


#Membres du groupe : Jules Mangeard ; Margot Schmutz ; Pierre Launay ; Eliot Nehme

#Projet : Isolation-based Anomaly Detection
#Jeu de données : fretelematic

##Importation des données

install.packages("xts")

install.packages(
  "CASdatasets",
  repos = "https://cas.uqam.ca/pub/",
  type = "source"
)

library(openxlsx)
library(CASdatasets)


data("fretelematic")

write.xlsx(fretelematic,
           "chemin_dossier_choisi"
           )




