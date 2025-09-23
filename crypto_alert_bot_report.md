# Bot de surveillance et d’alerte crypto – synthèse technique

## Objectifs et périmètre

L’objectif est de développer un bot Telegram capable de surveiller en continu les principaux marchés crypto (BTC/USDT, ETH/USDT, etc.) et d’alerter uniquement lorsque plusieurs indicateurs convergent vers un retournement haussier ou baissier.  Les règles de détection fournies par l’utilisateur ont été intégrées : croisements de moyennes mobiles, RSI, MACD, volume et indice de sentiment (Fear & Greed).  Un code Python complet est proposé dans le fichier `crypto_alert_bot.py`.

## Sources de données

* **Données de prix et volumes Binance** : le bot interroge directement l’API publique de Binance via l’endpoint *klines*. Ces données ne nécessitent aucune clé et fournissent des chandeliers OHLCV sur la période souhaitée.
* **Indice Fear & Greed** : Alternative.me met à disposition un endpoint public `https://api.alternative.me/fng/` qui retourne la valeur la plus récente de l’indice.  L’API accepte des paramètres facultatifs (nombre de valeurs, format et format de date) et renvoie un champ `value` (0 = extrême peur ; 100 = extrême avidité)【239967436881371†L64-L79】.  La documentation stipule que l’attribution est obligatoire et que l’utilisation commerciale est permise avec mention de la source【239967436881371†L208-L248】.
* **Données globales (facultatif)** : CoinMarketCap fournit l’endpoint `/v1/global-metrics/quotes/latest` qui retourne la capitalisation totale du marché, le volume global et la dominance BTC/ETH【256771951794996†L3631-L3638】.  Pour utiliser ces données un abonnement (clé API) est nécessaire.  Le service concurrent CoinGecko offre également un accès gratuit mais limité à environ 30 requêtes par minute【675826768962597†L120-L123】.
* **Autres API possibles** : la librairie `ccxt` propose une interface unifiée pour interroger plus de 100 plateformes d’échange【860877663968419†L18-L23】.  Elle n’est pas utilisée dans la version actuelle du bot mais peut être intégrée pour étendre la couverture à d’autres plateformes.

## Indicateurs techniques intégrés

Le script calcule les indicateurs en Python sans utiliser de bibliothèques externes de trading. Les séries sont dérivées des données OHLCV :

| Indicateur | Description | Règle déclenchant un signal |
|---|---|---|
| **MA50 / MA200** | Moyennes mobiles simples sur 50 et 200 périodes | *Golden Cross* : MA50 croise au-dessus de MA200.  *Death Cross* : MA50 croise sous MA200.  Le prix au‑dessus de MA200 indique une tendance haussière de long terme. |
| **RSI (14)** | Indicateur de momentum mesurant la force des mouvements | RSI > 55 : momentum haussier confirmé.  RSI < 45 : momentum baissier confirmé.  Croisement de 50 : bascule de tendance moyen/long terme. |
| **MACD** | Calculé à partir des EMA 12/26/9 ; histogramme = MACD – signal | MACD > signal : tendance haussière; MACD < signal : tendance baissière. |
| **Volume** | Somme des échanges; moyenne mobile sur 20 périodes | Volume actuel > moyenne : confirme le signal.  Volume faible : ignore le signal. |
| **Fear & Greed** | Indice sentiment, 0 – 100【239967436881371†L64-L79】 | < 25 : opportunité d’accumulation; > 75 : risque de retournement baissier. |

## Fonctionnement du bot

* **Récupération des données** : Pour chaque paire et chaque timeframe (1h, 4h, 1 jour et 1 semaine), le bot télécharge jusqu’à 300 chandeliers via l’API Binance.  Les colonnes `close`, `volume`, `open_time` et `close_time` sont converties dans un `DataFrame`.
* **Calcul des indicateurs** : Des fonctions dédiées calculent les SMA, le RSI, le MACD et la moyenne de volume.  Le code n’utilise pas de librairie de trading afin de rester autonome et lisible.
* **Détection des signaux** : la dernière bougie est comparée à l’avant‑dernière pour détecter les croisements.  Des règles combinent les conditions (par exemple RSI > 55 + MACD haussier + volume élevé) pour définir un signal haussier de court terme; l’alignement de trois conditions ou plus qualifie un signal **fort**, deux conditions un signal **moyen**, une seule un signal **faible**.
* **Index Fear & Greed** : lors de chaque scan, l’indice est récupéré.  Une valeur < 25 ajoute un argument haussier contrarien, alors qu’une valeur > 75 ajoute un argument baissier.
* **Envoi de l’alerte** : si un signal inédit est détecté, le bot assemble un message (avec emoji, paire, timeframe, heure de clôture, liste des raisons, indice Fear & Greed et, si disponible, capitalisation totale et dominance BTC) et l’envoie via l’API Telegram.
* **Persistance** : les signaux envoyés sont enregistrés dans une base SQLite afin d’éviter les doublons lors des scans suivants.
* **Fréquence des scans** : configurable dans `BotConfig.check_interval_seconds`.  Par défaut le script exécute un scan toutes les heures; il est possible de paramétrer des fréquences différentes selon vos besoins.

## Mise en place

1. **Cloner le dépôt ou copier les fichiers** dans un environnement Python 3.9+ équipé de `pip`.
2. **Installer les dépendances** : `pip install requests python-telegram-bot pandas`.
3. **Créer et configurer un bot Telegram** avec `@BotFather`, récupérer le **token** et l’**ID du chat** (privé ou groupe) et les renseigner dans `load_default_config()` ou via des variables d’environnement.
4. *(Optionnel)* **Créer un compte CoinMarketCap** si vous souhaitez enrichir les messages avec la capitalisation globale et la dominance BTC.  Renseignez la clé API dans le champ `cmc_api_key`.
5. **Lancer le script** : `python crypto_alert_bot.py`.  Le programme démarre une boucle infinie, effectue un scan puis attend l’intervalle configuré avant de recommencer.

## Points d’attention et améliorations possibles

* **Gestion des limites d’appels** : si vous décidez d’utiliser d’autres API (CoinGecko, Glassnode, etc.), veillez à respecter leurs limitations.  La version publique de CoinGecko ne permet qu’une trentaine de requêtes par minute【675826768962597†L120-L123】.
* **Extensibilité** : la librairie `ccxt` pourrait être introduite pour unifier l’accès à d’autres exchanges et récupérer les données on‑chain.  Cette librairie offre une API unifiée pour plus de 100 plateformes【860877663968419†L18-L23】.
* **Backtesting et optimisation** : pour affiner les règles, il serait pertinent d’intégrer un module de backtesting afin de mesurer les performances passées et ajuster les seuils.  L’architecture est conçue pour qu’il soit facile d’ajouter de nouveaux indicateurs (par exemple Ichimoku, Bollinger Bands, etc.).
* **Interface web ou email** : le cahier des charges mentionne la possibilité d’un tableau de bord ou de notifications par e‑mail.  Cela peut être ajouté en interfaçant le code avec un framework web léger (Flask/Django) ou un service comme SendGrid.

## Conclusion

Le bot présenté répond au cahier des charges en s’appuyant sur des données fiables : Binance pour les prix et volumes, Alternative.me pour l’indice Fear & Greed et CoinMarketCap pour les métriques globales.  Les règles de détection ont été codées de manière transparente, et l’architecture est modulaire pour faciliter les évolutions futures.