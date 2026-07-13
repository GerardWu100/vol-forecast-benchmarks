---
title: "Quand un benchmark de volatilité ne produit aucune prévision"
description: "Audit d'un benchmark de variance réalisée sur SPY, sans fuite d'information, puis analyse de sensibilité avec une fenêtre initiale d'un an."
date: 2026-07-13
image: images/cover.png
categories: ["Quantitative Research", "Risk Management"]
---

Un classement de modèles rassure. Pourtant, ce n'est pas le bon point de départ.

J'ai lancé le pipeline hors ligne du dépôt sans modifier sa configuration, depuis les barres à la minute jusqu'à l'évaluation walk-forward. Les deux premières étapes de calcul ont fonctionné : 564 lignes de variance réalisée quotidienne ont donné 537 lignes prêtes pour la modélisation, du 2022-11-02 au 2024-12-20. L'étape 4 n'a ensuite produit ni prévision ni score.

Ce résultat vide est correct. La configuration exige cinq années civiles d'apprentissage initial, alors que la matrice de variables couvre un peu plus de deux ans. Il n'existe donc aucune première date de prévision valide. Raccourcir discrètement la période d'apprentissage donnerait une démonstration plus flatteuse, mais un benchmark moins honnête.

Je suis les données jusqu'à cette limite. Je lance ensuite une analyse de sensibilité distincte avec une fenêtre d'un an, sans toucher à la configuration du dépôt, afin d'étudier la comparaison que l'échantillon par défaut ne permet pas.

## Que cherche-t-on à prévoir ?

Soit $P_{t,i}$ le cours de clôture de la minute $i$ pendant la séance $t$. Le rendement logarithmique intrajournalier vaut

$$
r_{t,i}=\log\left(\frac{P_{t,i}}{P_{t,i-1}}\right).
$$

Si la séance $t$ contient $M_t$ rendements intrajournaliers, la variance réalisée close-to-close est

$$
RV_t=\sum_{i=1}^{M_t}r_{t,i}^2.
$$

Ici, $RV_t$ est une estimation quotidienne de la variance, exprimée en rendement décimal au carré. Le pipeline conserve une séance seulement si elle contient au moins 300 barres à la minute. Il calcule aussi l'estimateur de Parkinson, fondé sur l'amplitude, et celui de Garman-Klass, fondé sur les cours d'ouverture, haut, bas et clôture. La cible reste néanmoins la variance réalisée close-to-close.

À la date de variables $t$, la cible à un jour est

$$
y_{t,1}=RV_{t+1},
$$

et la cible à cinq jours correspond à la moyenne des cinq observations suivantes,

$$
y_{t,5}=\frac{1}{5}\sum_{j=1}^{5}RV_{t+j}.
$$

Le graphique convertit la variance quotidienne en volatilité annualisée selon $100\sqrt{252RV_t}$, avec 252 séances par an. La courbe pâle est naturellement bruitée. Sa moyenne mobile sur 22 jours fait mieux ressortir la persistance.

![Volatilité réalisée annualisée de SPY et moyenne mobile sur 22 jours](images/01_realised_volatility.png)

La volatilité se regroupe par épisodes. Les observations calmes et agitées arrivent par blocs, et non comme des tirages indépendants. Une séparation aléatoire entre apprentissage et test mélangerait ces épisodes. Le projet apprend donc sur le passé, puis prévoit le bloc chronologique suivant.

## Trois horizons dans une même régression

Le modèle Heterogeneous Autoregressive Realised Variance, ou HAR-RV, représente la persistance à l'aide de résumés quotidien, hebdomadaire et mensuel. Définissons les trois variables disponibles à la date $t$ :

$$
x_{d,t}=RV_{t-1},
$$

$$
x_{w,t}=\frac{1}{5}\sum_{j=1}^{5}RV_{t-j},
$$

et

$$
x_{m,t}=\frac{1}{22}\sum_{j=1}^{22}RV_{t-j}.
$$

Pour un horizon $h$, exprimé en séances, la régression est

$$
\widehat y_{t,h}=\beta_0+\beta_d x_{d,t}+\beta_w x_{w,t}+\beta_m x_{m,t},
$$

où $\widehat y_{t,h}$ désigne la variance réalisée prévue et $\beta_0$, $\beta_d$, $\beta_w$ et $\beta_m$ les coefficients estimés.

Le décalage apparaît directement dans le code. La fenêtre mobile n'est construite qu'après avoir décalé la série de variance d'une ligne :

```python
shifted_series = lagged_df[column].shift(1)
rolling_mean = shifted_series.rolling(window=lag_window, min_periods=lag_window).mean()
lagged_df[lag_column_name] = rolling_mean
```

Le benchmark ajoute quatre solutions. Le modèle Generalized Autoregressive Conditional Heteroskedasticity avec un retard de rendement et un retard de variance, noté GARCH(1,1), modélise la variance conditionnelle à partir des rendements. Les régressions Ridge et Lasso sont ajustées sur le logarithme de la variance. Le dernier modèle est un ensemble d'arbres XGBoost. Ridge et Lasso utilisent les 14 variables, parmi lesquelles l'historique de variance réalisée, la volatilité implicite des options, le skew, l'écart entre volatilités implicite et réalisée, ainsi que le VIX.

## Le calendrier d'information fait partie du modèle

Une formule exacte peut malgré tout produire un backtest trompeur si les horodatages sont faux. Le pipeline considère qu'une chaîne d'options ou une clôture du VIX observée en $t-1$ devient disponible pour la ligne de variables en $t$. Il impose cette convention en avançant chaque observation d'un jour ouvré avant la jointure :

```python
vix_lagged["date"] = pd.to_datetime(vix_lagged["date"]) + pd.offsets.BDay(1)
vix_lagged = vix_lagged.sort_values("date").reset_index(drop=True)
```

Le même décalage s'applique aux variables d'options. Les courtes interruptions peuvent être comblées par propagation de la dernière valeur pendant trois jours au maximum. Le pipeline supprime ensuite les lignes auxquelles il manque une variable ou une cible requise. Ces règles font passer l'échantillon de 564 lignes de variance réalisée à 537 lignes complètes.

L'évaluation suit une fenêtre croissante. Après la période initiale d'apprentissage, chaque modèle prévoit les 21 séances suivantes. L'échantillon d'apprentissage s'allonge, le modèle est réestimé, puis le cycle recommence. Aucune donnée du bloc de test n'entre dans l'estimation qui le précède.

## Pourquoi le classement configuré est vide

L'échantillon de variables commence le 2022-11-02. L'évaluateur ajoute les cinq années d'apprentissage prévues par la configuration, ce qui place la première coupure possible en novembre 2027. Or la dernière ligne est datée du 2024-12-20.

| Audit de l'exécution par défaut | Valeur |
|---|---:|
| Lignes de variance réalisée quotidienne | 564 |
| Lignes complètes de variables | 537 |
| Période couverte | du 2022-11-02 au 2024-12-20 |
| Période initiale d'apprentissage configurée | 5 années civiles |
| Lignes de prévision | 0 |
| Lignes de score | 0 |

L'évaluateur vérifie que cette coupure existe. Dans le cas contraire, il renvoie une table de prévisions vide avec le bon schéma. Ce comportement vaut mieux qu'un apprentissage sur un historique raccourci sans le signaler.

Il reste un problème de configuration. Un benchmark portable devrait fournir assez de données pour exécuter ses paramètres par défaut, ou adopter des paramètres compatibles avec les données livrées. Dans son état actuel, le dépôt reproduit la construction des variables, mais pas la comparaison de modèles annoncée.

## Une analyse de sensibilité sur un an

Pour examiner le moteur d'évaluation, j'ai copié la configuration chargée en mémoire et modifié un seul champ dans cette copie :

```python
sensitivity_config = deepcopy(config)
sensitivity_config["forecast"]["initial_train_years"] = 1
```

La liste des modèles, la réestimation tous les 21 jours, les variables, les cibles et le code de scoring sont restés identiques. J'obtiens ainsi 286 prévisions hors échantillon pour chaque modèle et chaque horizon.

Le score principal est la perte de quasi-vraisemblance, ou QLIKE. Pour une variance réalisée $y_t>0$ et une variance prévue $\widehat y_t>0$, la perte par observation est

$$
L_t=\log(\widehat y_t)+\frac{y_t}{\widehat y_t}.
$$

Une perte moyenne plus faible est préférable lorsque les modèles prévoient la même série cible. Les valeurs absolues sont négatives, car les variances sont exprimées en petites unités décimales. Leur ordre et leurs écarts sont les éléments pertinents.

| Horizon | Modèle | QLIKE | QLIKE vs. HAR-RV | MSE | Observations |
|---:|---|---:|---:|---:|---:|
| 1 day | Lasso | -8.8272 | -0.0301 | 5.9584e-09 | 286 |
| 1 day | Ridge | -8.8268 | -0.0296 | 6.2964e-09 | 286 |
| 1 day | HAR-RV | -8.7972 | 0.0000 | 2.8105e-09 | 286 |
| 1 day | GARCH(1,1) | -8.7519 | 0.0453 | 3.1568e-09 | 286 |
| 1 day | XGBoost | -8.6919 | 0.1052 | 3.9683e-09 | 286 |
| 5 days | Lasso | -8.8008 | -0.0309 | 2.3718e-09 | 286 |
| 5 days | Ridge | -8.7967 | -0.0268 | 2.2992e-09 | 286 |
| 5 days | HAR-RV | -8.7699 | 0.0000 | 1.6964e-09 | 286 |
| 5 days | GARCH(1,1) | -8.7471 | 0.0228 | 1.7151e-09 | 286 |
| 5 days | XGBoost | -8.6966 | 0.0733 | 2.4243e-09 | 286 |

![Écarts de QLIKE par rapport à HAR-RV dans l'analyse sur un an](images/02_qlike_vs_har.png)

Lasso obtient le QLIKE le plus faible aux deux horizons, juste devant Ridge. HAR-RV affiche en revanche la plus faible erreur quadratique moyenne, ou MSE, dans les deux cas. Il n'y a pas de contradiction : la MSE pondère les erreurs absolues au carré, tandis que QLIKE pénalise le rapport entre variance réalisée et variance prévue. Le choix de la fonction de perte détermine le type d'erreur que l'on juge le plus coûteux.

## Comparaison statistique et santé des prévisions

Le projet calcule aussi des tests de Diebold-Mariano par paire. Pour les modèles $a$ et $b$, définissons l'écart de perte

$$
d_t=L_{a,t}-L_{b,t}.
$$

L'hypothèse nulle est $E[d_t]=0$. Une statistique positive indique que le modèle $a$ subit une perte moyenne supérieure à celle du modèle $b$. Dans l'analyse de sensibilité, la statistique HAR-RV contre Lasso atteint 2.0812, avec une $p$-value bilatérale de 0.0374 à un jour. À cinq jours, elle vaut 2.1698 et la $p$-value 0.0300. Selon l'approximation asymptotique utilisée ici, les deux comparaisons rejettent l'égalité des précisions QLIKE au seuil de 5%.

Cette phrase appelle de la prudence. Les horizons à cinq jours se chevauchent, l'échantillon ne contient qu'un actif et le résultat provient d'une fenêtre initiale raccourcie après constat. Une $p$-value ne corrige pas une spécification de recherche fragile.

Les diagnostics apportent tout de même une information rassurante : aucune des 2,860 prévisions de sensibilité n'atteint le plancher de variance positive. La transformation logarithmique de la cible dans les modèles linéaires évite ici un écrêtage qui pourrait donner une fausse impression de stabilité.

## Ce que je modifierais ensuite

Je commencerais par réconcilier la configuration par défaut avec les données portables. Conserver une fenêtre initiale de cinq ans suppose de fournir un historique nettement plus long. Conserver l'historique actuel suppose de choisir et documenter une fenêtre plus courte avant d'observer le classement.

Je répéterais ensuite le benchmark sur plusieurs sous-jacents liquides et sur divers régimes de volatilité. SPY accompagné du VIX constitue une démonstration utile, mais ne prouve pas la supériorité générale d'une famille de modèles. La fréquence de réestimation et la longueur de la fenêtre initiale devraient aussi faire l'objet de tests de sensibilité définis à l'avance.

Enfin, j'examinerais les variables d'options selon leur horodatage et leur couverture à la source, avant même leur entrée dans la matrice de régression. Le décalage d'un jour ouvré empêche une fuite directe, mais des cotations anciennes, des échéances rares ou la propagation limitée des dernières valeurs peuvent encore changer le sens économique des variables.

Le résultat principal n'est donc pas la victoire de Lasso. C'est le refus, par un évaluateur correctement conçu, de fabriquer un échantillon de test. L'analyse de sensibilité montre que le pipeline sait comparer les modèles dès que la fenêtre devient réalisable. L'exécution vide montre que la spécification de recherche doit encore être mise en accord avec ses données.
