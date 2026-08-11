---
title: "Prévoir la variance de SPY sans tricher avec le calendrier"
description: "Comparaison sans fuite d'information de HAR-RV, GARCH, régressions régularisées et XGBoost, avec un protocole walk-forward qui refuse les historiques d'apprentissage impossibles."
date: 2026-07-13
image: images/cover.png
categories: ["Quantitative Research", "Risk Management"]
---

La première version de ce benchmark se terminait sans erreur, mais ne produisait aucune prévision. La matrice de variables s'arrêtait en décembre 2024, alors que la configuration réservait cinq années civiles à l'apprentissage initial à partir de novembre 2022. La première date de test possible se serait donc située en novembre 2027.

Une table de scores vide était cohérente sur le plan mathématique, mais trompeuse sur le plan opérationnel. Lorsqu'un pipeline écrit des fichiers, affiche « complete » dans ses logs et renvoie le code de sortie zéro, il paraît sain, même si la question de recherche n'a jamais été testée.

J'ai corrigé ce contrat à deux endroits. La configuration portable utilise désormais une année civile d'apprentissage initial, compatible avec les données fournies et laissant 286 dates hors échantillon. Une fenêtre plus longue reste permise, mais l'étape 4 lève une exception `InsufficientTrainingHistoryError` accompagnée des dates lorsque l'historique est insuffisant. On obtient ainsi un benchmark exécutable dont les limites apparaissent avant le moindre classement.

## Ce que l'on cherche à prévoir

Les données contiennent des barres à la minute pour SPY, le ticker du fonds négocié en bourse SPDR S&P 500, ainsi que des chaînes d'options et le Cboe Volatility Index (VIX). La cible est la variance réalisée, et non la volatilité implicite des options ni un rendement de trading.

Soit $P_{t,i}$ le cours de clôture de la minute $i$ pendant la séance $t$. Le rendement logarithmique intrajournalier vaut

$$
r_{t,i}=\log\left(\frac{P_{t,i}}{P_{t,i-1}}\right).
$$

Si la séance $t$ contient $M_t$ rendements intrajournaliers, la variance réalisée close-to-close est la somme de leurs carrés :

$$
RV_t=\sum_{i=1}^{M_t}r_{t,i}^2.
$$

Ici, $RV_t$ est exprimée en rendement décimal au carré. Le pipeline exige au moins 300 barres à la minute par séance. Il calcule aussi l'estimateur de Parkinson, fondé sur l'amplitude, et celui de Garman-Klass, fondé sur les cours d'ouverture, haut, bas et clôture. Cependant, $RV_t$ fournit la cible et les principales variables du modèle Heterogeneous Autoregressive Realised Variance (HAR-RV). Cette construction s'inscrit dans les travaux sur la volatilité réalisée d'[Andersen, Bollerslev, Diebold et Labys](https://doi.org/10.1111/1468-0262.00418).

Pour une ligne de variables datée $t$, la cible à un jour est

$$
y_{t,1}=RV_{t+1}.
$$

La cible à cinq jours est la moyenne arithmétique des cinq variances quotidiennes suivantes :

$$
y_{t,5}=\frac{1}{5}\sum_{j=1}^{5}RV_{t+j}.
$$

Le graphique convertit la variance en volatilité annualisée selon $100\sqrt{252RV_t}$, avec 252 séances par an. Cette racine carrée sert uniquement à faciliter la lecture du graphique. Les modèles et les fonctions de perte restent exprimés en variance.

![Volatilité réalisée annualisée de SPY et moyenne mobile sur 22 jours](images/01_realised_volatility.png)

La série quotidienne pâle connaît des sauts marqués, tandis que sa moyenne sur 22 jours évolue par blocs persistants. C'est le regroupement de volatilité. Une séparation aléatoire entre apprentissage et test serait donc mal adaptée : des observations appartenant au même épisode se retrouveraient des deux côtés.

## Que sait-on à chaque date de prévision ?

Le calendrier des variables détermine si l'exercice constitue une prévision ou une reconstruction. Sur la ligne $t$, les variables de variance réalisée contiennent les observations jusqu'à $t-1$. La composante quotidienne de HAR est $RV_{t-1}$, tandis que les composantes hebdomadaire et mensuelle moyennent les 5 et 22 séances précédentes.

Le code applique le décalage avant la fenêtre mobile :

```python
shifted_series = lagged_df[column].shift(1)
rolling_mean = shifted_series.rolling(window=lag_window, min_periods=lag_window).mean()
lagged_df[lag_column_name] = rolling_mean
```

Les variables tirées des options et le VIX sont eux aussi avancés d'un jour ouvré avant leur jointure avec la ligne $t$. Les variables d'options sont la volatilité implicite at-the-money (ATM IV), le skew put-call à delta 25, la pente de la structure par terme et l'écart entre volatilité implicite et volatilité réalisée annualisée. Les courtes interruptions peuvent être comblées par propagation de la dernière valeur pendant trois jours au maximum.

Le modèle GARCH fait exception à la convention $t-1$. Il utilise le rendement quotidien observé à la date $t$ pour prévoir la variance à partir de $t+1$. C'est valide pour une prévision effectuée en fin de séance, mais le benchmark combine alors deux ensembles d'information : l'état du marché et des options retardé pour les régressions, et le rendement close-to-close courant pour GARCH. Cette différence prudente ne crée pas de fuite vers le futur, mais elle complique une comparaison parfaitement homogène.

Après filtrage des lignes incomplètes, le pipeline conserve 537 lignes du 2022-11-02 au 2024-12-20. Les 251 premières constituent l'échantillon initial d'une année civile. L'évaluation commence le 2023-11-02.

## Cinq façons de prévoir la variance

### HAR-RV : la persistance à trois échelles

Le modèle HAR-RV proposé par [Corsi](https://doi.org/10.1093/jjfinec/nbp001) résume la persistance au moyen de moyennes quotidienne, hebdomadaire et mensuelle. Définissons

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

Pour un horizon $h$, exprimé en séances, les moindres carrés ordinaires estiment

$$
\widehat y_{t,h}
=\beta_0+\beta_d x_{d,t}+\beta_w x_{w,t}+\beta_m x_{m,t},
$$

où $\widehat y_{t,h}$ est la variance prévue, tandis que $\beta_0$, $\beta_d$, $\beta_w$ et $\beta_m$ sont les coefficients estimés. Les prévisions négatives sont ramenées à un petit plancher positif, car QLIKE exige des valeurs strictement positives.

### GARCH(1,1) : la variance conditionnelle tirée des rendements

Le modèle Generalized Autoregressive Conditional Heteroskedasticity avec un retard de choc et un retard de variance, abrégé GARCH(1,1), suit [Bollerslev](https://doi.org/10.1016/0304-4076(86)90063-1). Le modèle de rendement de moyenne nulle est

$$
r_t=\sigma_t\varepsilon_t,
$$

où $r_t$ est le rendement quotidien, $\sigma_t>0$ l'écart-type conditionnel et $\varepsilon_t$ un choc de moyenne nulle et de variance unitaire. La variance conditionnelle évolue selon

$$
\sigma_{t+1}^2
=\omega+\alpha r_t^2+\beta\sigma_t^2,
$$

où $\omega>0$ est la constante de variance, $\alpha\geq0$ la réaction au choc et $\beta\geq0$ la persistance de la variance. Au-delà d'un jour, les chocs futurs attendus vérifient $E_t[r_{t+k}^2]=E_t[\sigma_{t+k}^2]$. Le code poursuit donc la récursion avec la persistance $\alpha+\beta$, puis moyenne les $h$ prochaines variances conditionnelles afin de correspondre à $y_{t,h}$ :

```python
one_step_variance = (
    omega_decimal + self._alpha * observed_return**2 + self._beta * previous_variance
)

horizon_variance = one_step_variance
variance_sum = one_step_variance
for _ in range(1, self._forecast_horizon):
    horizon_variance = omega_decimal + persistence * horizon_variance
    variance_sum += horizon_variance
```

Cette récursion dépendante de l'horizon est nécessaire. Réutiliser une variance conditionnelle à un jour pour la cible moyenne à cinq jours reviendrait à répondre à une autre question.

### Ridge et Lasso : régulariser dans l'espace de la log-variance

Soit $\mathbf{x}_t\in\mathbb{R}^{14}$ le vecteur de variables standardisées et $z_{t,h}=\log(y_{t,h})$ la cible en log-variance. Ridge estime le vecteur de coefficients $\boldsymbol\theta$ en résolvant

$$
\widehat{\boldsymbol\theta}_{\mathrm{ridge}}
=\arg\min_{\boldsymbol\theta}
\left\{
\sum_{t=1}^{T}\left(z_{t,h}-\theta_0-\mathbf{x}_t^\top\boldsymbol\theta\right)^2
+\lambda\lVert\boldsymbol\theta\rVert_2^2
\right\},
$$

où $T$ est le nombre de lignes d'apprentissage, $\lambda\geq0$ contrôle le rétrécissement et $\lVert\boldsymbol\theta\rVert_2^2$ désigne la somme des carrés des coefficients. Le Lasso, d'après [Tibshirani](https://doi.org/10.1111/j.2517-6161.1996.tb02080.x), remplace cette pénalité quadratique par la somme des valeurs absolues :

$$
\widehat{\boldsymbol\theta}_{\mathrm{lasso}}
=\arg\min_{\boldsymbol\theta}
\left\{
\sum_{t=1}^{T}\left(z_{t,h}-\theta_0-\mathbf{x}_t^\top\boldsymbol\theta\right)^2
+\lambda\lVert\boldsymbol\theta\rVert_1
\right\}.
$$

Le paramètre de pénalité $\lambda$ est choisi par validation croisée temporelle. L'exponentielle ramène ensuite la prévision vers une variance positive :

```python
positive_predictions = np.exp(log_predictions)
return np.maximum(positive_predictions, MIN_POSITIVE_VARIANCE)
```

L'ajustement en espace logarithmique change la cible de la régression et donc la géométrie des erreurs. Ce n'est pas un simple artifice numérique.

### XGBoost : des interactions non linéaires

XGBoost ajuste un ensemble additif d'arbres de régression, suivant [Chen et Guestrin](https://doi.org/10.1145/2939672.2939785). Avec les fonctions d'arbre $f_m$ et le taux d'apprentissage $\eta$, la prévision après $M$ arbres s'écrit

$$
F_M(\mathbf{x}_t)=\sum_{m=1}^{M}\eta f_m(\mathbf{x}_t).
$$

Le projet emploie 200 arbres d'une profondeur maximale de 4, un sous-échantillonnage des lignes et colonnes de 0.8, ainsi qu'une graine aléatoire fixe. Contrairement à HAR-RV ou GARCH, les arbres n'imposent aucune équation de volatilité. Ils cherchent des seuils et des interactions non linéaires dans la même matrice de 14 variables.

## Un protocole walk-forward qui sait échouer honnêtement

La configuration portable corrigée réserve une année civile à l'apprentissage initial. Chaque modèle prévoit ensuite le bloc suivant de 21 séances. À la coupure suivante, l'échantillon d'apprentissage s'élargit à toutes les lignes déjà observées et le modèle est réestimé. Une date de prévision n'entre jamais dans l'ajustement qui la précède.

| Audit de l'exécution par défaut | Valeur |
|---|---:|
| Lignes de variance réalisée | 564 |
| Lignes complètes de variables | 537 |
| Période des variables | 2022-11-02 to 2024-12-20 |
| Période initiale d'apprentissage | 1 calendar year |
| Première date de prévision | 2023-11-02 |
| Dernière date de prévision | 2024-12-20 |
| Origines de prévision par modèle et horizon | 286 |

Une année n'est pas présentée comme une durée d'estimation optimale. C'est un choix adapté à l'échantillon portable, fixé avant la comparaison des scores. Une étude avec cinq années initiales exige davantage de données historiques.

L'ancien code renvoyait une table vide lorsqu'une fenêtre était impossible. L'évaluateur s'arrête désormais :

```python
raise InsufficientTrainingHistoryError(
    f"{symbol} horizon={horizon}: initial training window is infeasible. "
    f"The {len(working_df)} feature rows span {first_date.date()} to "
    f"{last_date.date()}, while initial_train_years={initial_train_years} "
    f"requires a first evaluation date on or after {required_cutoff.date()}"
)
```

Cet arrêt protège le sens de toutes les tables produites ensuite. Un fichier vide peut être un artefact logiciel valide, mais il ne constitue pas une preuve empirique.

## Deux pertes, deux conceptions d'une bonne prévision

Soit $n$ le nombre d'observations hors échantillon, $y_t>0$ la variance réalisée et $\widehat y_t>0$ la variance prévue. L'erreur quadratique moyenne, ou MSE, est

$$
\operatorname{MSE}
=\frac{1}{n}\sum_{t=1}^{n}\left(y_t-\widehat y_t\right)^2.
$$

La MSE s'exprime en unités de variance au carré et donne beaucoup de poids aux grandes erreurs absolues. La perte de quasi-vraisemblance utilisée par le projet, ou QLIKE, est

$$
\operatorname{QLIKE}
=\frac{1}{n}\sum_{t=1}^{n}
\left[
\log(\widehat y_t)+\frac{y_t}{\widehat y_t}
\right].
$$

QLIKE s'intéresse au rapport entre variance réalisée et variance prévue. Elle appartient à la classe de fonctions de perte étudiée par [Patton](https://doi.org/10.1016/j.jeconom.2010.03.034) en présence de proxys de volatilité bruités. Pour les deux métriques, une valeur plus faible est préférable. Les QLIKE sont négatifs ici, car la variance est enregistrée dans de petites unités décimales. Un changement d'unité décale leur niveau absolu. L'information réside donc dans les comparaisons menées sur une même cible et dans les mêmes unités.

## Résultats corrigés de la configuration portable

La commande portable, exécutée sans modification, produit maintenant les scores suivants :

| Horizon | Modèle | QLIKE | QLIKE vs. HAR-RV | MSE | Observations |
|---:|---|---:|---:|---:|---:|
| 1 day | Lasso | -8.8272 | -0.0301 | 5.9584e-09 | 286 |
| 1 day | Ridge | -8.8268 | -0.0296 | 6.2964e-09 | 286 |
| 1 day | HAR-RV | -8.7972 | 0.0000 | 2.8105e-09 | 286 |
| 1 day | GARCH(1,1) | -8.7618 | 0.0353 | 3.1336e-09 | 286 |
| 1 day | XGBoost | -8.6919 | 0.1052 | 3.9683e-09 | 286 |
| 5 days | Lasso | -8.8008 | -0.0309 | 2.3718e-09 | 286 |
| 5 days | Ridge | -8.7967 | -0.0268 | 2.2992e-09 | 286 |
| 5 days | HAR-RV | -8.7699 | 0.0000 | 1.6964e-09 | 286 |
| 5 days | GARCH(1,1) | -8.7472 | 0.0226 | 1.7717e-09 | 286 |
| 5 days | XGBoost | -8.6966 | 0.0733 | 2.4243e-09 | 286 |

![Écarts de QLIKE par rapport à HAR-RV pour la configuration portable](images/02_qlike_vs_har.png)

Lasso obtient le QLIKE le plus faible aux deux horizons, suivi de près par Ridge. HAR-RV affiche la plus faible MSE dans les deux cas. À un jour, la MSE de Lasso dépasse même le double de celle de HAR-RV, malgré sa victoire selon QLIKE. Parler du « meilleur modèle » sans nommer la fonction de perte est donc incomplet.

![Volatilité réalisée à un jour et prévisions de HAR-RV et Lasso](images/03_forecast_paths.png)

Les deux chemins de prévision lissent les mouvements les plus brusques de la volatilité réalisée. Lasso peut produire des prévisions isolées plus élevées, ce qui dégrade son erreur quadratique lorsque le mouvement ne se réalise pas. Son QLIKE inférieur indique un meilleur rapport de variance en moyenne, et non des erreurs plus petites à chaque date.

Aucune des 2,860 prévisions n'atteint le plancher de variance positive. Ce diagnostic compte : un écrêtage fréquent peut donner une impression de stabilité tout en masquant des prévisions brutes négatives et invalides.

## Ce qu'apportent les tests par paire

Pour les modèles $a$ et $b$, soient $L_{a,t}$ et $L_{b,t}$ leurs pertes QLIKE à la date $t$. L'écart de perte de Diebold-Mariano est

$$
d_t=L_{a,t}-L_{b,t}.
$$

L'hypothèse nulle est $E[d_t]=0$. Une statistique positive indique que le modèle $a$ subit une perte moyenne supérieure. Le test suit [Diebold et Mariano](https://doi.org/10.1080/07350015.1995.10524599), avec une estimation de variance de long terme de type Newey-West dans cette implémentation.

Pour HAR-RV contre Lasso, la statistique vaut 2.0812 avec une $p$-value bilatérale de 0.0374 à un jour. À cinq jours, elle atteint 2.1698 avec une $p$-value de 0.0300. Selon l'approximation asymptotique du code, ces deux comparaisons rejettent l'égalité des précisions QLIKE au seuil de 5%.

Ces décimales donnent une impression de précision supérieure à la portée réelle de l'échantillon. Les cibles à cinq jours se chevauchent. Dix tests par paire sont effectués à chaque horizon sans correction pour comparaisons multiples. L'échantillon ne contient qu'un actif et environ quatorze mois de prévisions hors échantillon. Il s'agit de diagnostics de benchmark, pas d'un classement universel des modèles de volatilité.

## Ce qui reste à résoudre

La configuration corrigée répond désormais à la question annoncée, mais quatre limites demeurent.

D'abord, une année d'apprentissage est courte pour un modèle censé traverser plusieurs régimes de volatilité. Il vaut mieux allonger l'historique brut que régler la fenêtre à partir de ces résultats. Ensuite, SPY et VIX ne permettent aucune conclusion générale entre actifs. Les variables d'options sont retardées correctement, mais des cotations anciennes, des échéances rares et la propagation sur trois jours peuvent encore modifier leur sens économique. Enfin, l'ensemble d'information n'est pas identique entre les familles, car GARCH emploie le rendement quotidien courant alors que les autres variables d'état du marché sont retardées.

Le résultat le plus solide concerne la méthode. Trois horloges doivent concorder dans un benchmark de prévision : l'horizon de la cible, la disponibilité des variables et la coupure de l'échantillon d'apprentissage. Une fois ces horloges explicites, le désaccord entre scores devient instructif. Lasso est meilleur selon QLIKE, HAR-RV selon la MSE, et aucune de ces conclusions ne dépasse cet échantillon sans données supplémentaires.

## Références primaires

- Andersen, T. G., Bollerslev, T., Diebold, F. X., and Labys, P. (2003). [Modeling and Forecasting Realized Volatility](https://doi.org/10.1111/1468-0262.00418).
- Bollerslev, T. (1986). [Generalized Autoregressive Conditional Heteroskedasticity](https://doi.org/10.1016/0304-4076(86)90063-1).
- Chen, T., and Guestrin, C. (2016). [XGBoost: A Scalable Tree Boosting System](https://doi.org/10.1145/2939672.2939785).
- Corsi, F. (2009). [A Simple Approximate Long-Memory Model of Realized Volatility](https://doi.org/10.1093/jjfinec/nbp001).
- Diebold, F. X., and Mariano, R. S. (1995). [Comparing Predictive Accuracy](https://doi.org/10.1080/07350015.1995.10524599).
- Patton, A. J. (2011). [Volatility Forecast Comparison Using Imperfect Volatility Proxies](https://doi.org/10.1016/j.jeconom.2010.03.034).
- Tibshirani, R. (1996). [Regression Shrinkage and Selection via the Lasso](https://doi.org/10.1111/j.2517-6161.1996.tb02080.x).
