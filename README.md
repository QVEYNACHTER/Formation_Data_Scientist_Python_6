Pour ce projet, on m'a chargé de plusieurs missions : Construire un modèle de scoring qui donnera une prédiction sur la probabilité de faillite d'un client de façon automatique, faire une analyse de feature importance globale et locale, mettre en production le modèle de scoring de prédiction à l’aide d’une API et réaliser une interface de test de cette API, mettre en oeuvre une approche globale MLOps de bout en bout, et détecter du Data Drift. Je travaille pour une institution financière qui souhaite mettre en oeuvre un outil de "scoring crédit" pour calculer la probabilité qu’un client rembourse son crédit, puis classifie la demande en crédit accordé ou refusé. Elle souhaite donc développer un algorithme de classification en s'appuyant sur des sources de données variées (données comportementales, données provenant d'autres institutions financières, etc.). Le Fichier Introductif dans le Dossier Code décrit la structure du repository.

Dans le Notebook Modélisation, je me suis d'abord focalisé sur le nettoyage des données (avec énormément de feature engineering), ainsi qu'une courte analyse exploratoire. Ensuite, j'ai enchaîné sur l'entraînement des modèles et le tracking MLFlow. La librairie MLFlow m'a permis de logger les paramètres des modèles, les métriques, le temps d'entraînement ainsi que les modèles eux-mêmes. J'ai utilisé l'AUC (Area Under Curve) et le recall comme métriques pour affiner les hyperparamètres des modèles. En effet, comme on cherche à minimiser le nombre de faux négatifs, le recall est la métrique la plus pertinente à suivre.

J'ai testé deux modèles : Une régression logistique (linéaire) et LightGBM (une version plus légère de XGBoost, non linéaire). Comme le dataset contient des NaNs, il a fallu procéder à une imputation avant la régression logistique. En revanche, LightGBM fonctionne bien sur des données qui contiennent des NaNs, donc je n'ai pas eu besoin de dénaturer le dataset avec une imputation.

Voici les points auxquels j'ai dû faire attention :
- Déséquilibre entre le nombre de bons et de moins bons clients.
- Déséquilibre du coût métier entre un faux négatif et un faux positif. On estime que le coût d'un faux négatif est 10 fois supérieur au coût d'un faux positif. D'où la création d'un score "métier" dans le but de minimiser le coût d'erreur de prédiction. Cela veut dire qu'un bon score métier est un score faible. Ce score permet de choisir le meilleur modèle et ses meilleurs hyperparamètres.
- Seuil de détermination des classes. Toujours dans l'optique de minimiser le score métier, "predict" suppose un seuil à 0.5, qui n'est pas forcément optimal.
- Score de l'AUC, un score supérieur à 0.82 (score du vainqueur du challenge Kaggle) pourrait suggérer de l'overfitting.

J'ai utilisé la librairie optuna car elle utilise une approche de recherche d'hyperparamètres appelée optimisation bayésienne. Au lieu de tester toutes les combinaisons possibles (comme GridSearchCV), optuna échantillonne les valeurs de manière adaptative, en se basant sur les performances des essais précédents, ce qui permet de converger plus rapidement vers des solutions optimales.

Une fois cela fait, dans le notebook Meilleurs modèles et feature importance (dans le Dossier Code), j'ai enchâiné sur l'entraînement des modèles avec les meilleurs hyperparamètres retenus précédemment, que j'ai également strocké sur le serveur MLFlow. Puis j'ai procédé à une analyse de feature importance globale grâce aux coefficients des modèles, et locale grâce à la librairie SHAP. Je me suis également intéressé à la distribution des prédictions correctes et incorrectes, pour me rendre compte que passé le threshold, la proportion de prédictions incorrectes montait en flèche. Cela est dû au déséquilibre du coût métier entre faux négatifs et faux positifs, pour minimiser les faux négatifs, on augmente inévitablement le nombre de faux positifs.

On passe désormais au déploiement :
- api.py est une application Flask dans laquelle j'ai défini une fonction pour effectuer une prédiction à partir d'un identifiant client
- dashboard.py est un dashboard streamlit qui permet de visualiser les résultats de la prédiction ainsi que la feature importance locale
- setup.sh est un script d'installation automatisée configurant l'environnement avant exécution de l'application
- requirements.txt liste les librairies Python nécessaires au déploiement
- runtime.txt définit la version de Python à utiliser
- Procfile définit les processus à exécuter lors du déploiement sur Heroku
- (Le premier livrable API sert pour l'exécution de l'API et du dashboard en local)
- Le cinquième livrable Script Test API sert aux test unitaires (3 API, 3 dashboard) avant déploiement (réalisables directement sur Heroku ou bien dans un workflow avec GitHub Actions)

Elle n'est plus active, mais il y avait une pipeline de déploiement connectée entre ce repository et Heroku.

On enchaîne avec l'analyse de Data Drift. Celle-ci consiste à détecter les changements dans la distribution des données entre deux ensembles. Dans mon cas, j'ai comparé les données utilisées jusqu'alors avec le jeu de test qui a été mis à l'écart après le premier notebook. J'ai utilisé la librairie evidently dans le notebook Data Drift (dans le Dossier Code), le résultat étant le quatrième livrable Tableau HTML Data Drift Evidently.

Il s'agit de loin du projet le plus long et le plus complexe sur lequel j'ai travaillé jusqu'à maitenant.
