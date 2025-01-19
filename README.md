Fonctionnalité:
Recherche de texte (toutes combinaisons confondues)
Streaming intelligent
Calcul de similarité en utilisant les méthodes de vectorisation (Word2Vec, FastText)
Implémentation de 2 chatbots:
Chatbot avec Word2Vec et TF-IDF
Chatbot en utilisant un LLM via API call
Améliorations et ajouts:
Implémentation du Nuage de Mots avec TF-IDF comme Spécifié sur Moodle:
Correction de la Recherche de Corpus dans d’autres Corpus:
Ajout des méthodes GloVe et Doc2Vec

Recherche de Texte (Toutes Combinaisons Confondues): 
Grâce à l'application, on peut chercher un corpus dans d'autres corpus, chercher un fichier (document) dans d'autres corpus, chercher des phrases dans des fichiers ou corpus, et bien d’autres combinaisons simplement en spécifiant ‘quoi chercher’ et ‘où chercher’. L'application permet de décrire nos propres phrases ou textes et permet aussi d'uploader nos propres fichiers ou corpus pour une customisation maximale. L’application permet également d’afficher les documents ou corpus les plus proches (k plus proches). Pour le calcul de similarité, je permet à l'utilisateur d'utiliser divers descripteurs et distances (tous ceux que l'on a vus en cours). Tout cela a été implémenté grâce à de multiples fonctions que vous trouverez dans le code.
Streaming Intelligent : 
Le système de "streaming intelligent" fonctionne en évaluant plusieurs méthodes de stemming pour sélectionner celle qui est la plus adaptée. Voici les étapes détaillées avec des formules simplifiées :
Entrées : On prend deux éléments :
Q : Le texte ou la phrase "quoi chercher".
C : Le texte ou corpus "où chercher".
Lemmatisation : On utilise SpaCy pour créer une version lemmatisée de Q et C :
Q_lemma = lemmatize(Q)
C_lemma = lemmatize(C)
Ces versions servent de référence sémantique.
Stemming : On applique trois algorithmes de stemming sur Q : Porter, Lancaster et Snowball.
Q_stem_Porter = stem_Porter(Q)
Q_stem_Lancaster = stem_Lancaster(Q)
Q_stem_Snowball = stem_Snowball(Q)
Similarité TF-IDF : On compare chaque version stemmée avec la version lemmatisée pour mesurer la préservation du sens. La similarité est calculée comme :
similarity_S = (Q_lemma * Q_stem_S) / (|Q_lemma| * |Q_stem_S|)
où * représente le produit scalaire des vecteurs TF-IDF, et |x| est la norme (longueur) du vecteur.
Ratio de réduction : On mesure la réduction en longueur de chaque méthode :
reduction_ratio_S = len(Q_stem_S) / len(Q)
Une pénalité est appliquée si le ratio est hors de l'intervalle [0.5, 0.9] :
adjusted_similarity_S = similarity_S * 0.8 (si reduction_ratio_S < 0.5 ou > 0.9) adjusted_similarity_S = similarity_S (sinon)
Comparaison avec le texte cible : Chaque version stemmer est comparée avec C_lemma pour évaluer son adéquation contextuelle. La similarité est calculée de la même manière :
context_similarity_S = (Q_stem_S * C_lemma) / (|Q_stem_S| * |C_lemma|)
Score final : Pour chaque méthode de stemming, le score final combine les similarités sémantique et contextuelle :
final_score_S = 0.5 * adjusted_similarity_S + 0.5 * context_similarity_S
Sélection de la meilleure méthode : Le système choisit la méthode avec le score final le plus élevé :
best_stemmer = max(final_score_S)
Ce système optimise à la fois la réduction linguistique et la pertinence contextuelle pour des recherches plus efficaces.
Calcul de Similarité en Utilisant les Méthodes de Vectorisation (Word2Vec, FastText): 
Dans l’application, lorsqu’un utilisateur choisit quoi chercher (par exemple, un mot ou une phrase) et où chercher (dans des dossiers, fichiers, ou des portions de texte), le système utilise des méthodes de vectorisation comme Word2Vec ou FastText pour effectuer un calcul de similarité selon la méthode que l'utilisateur choisie. Tout d’abord, le texte sélectionné est prétraité : il est nettoyé en supprimant les caractères spéciaux, les mots vides (stopwords) et en mettant tout en minuscules, avant d’être découpé en mots ou en phrases. Ensuite, chaque mot ou phrase est converti en vecteur numérique à l’aide de modèles Word2Vec ou FastText, qui représentent chaque mot dans un espace vectoriel où les mots au sens similaire sont proches les uns des autres. Pour une phrase ou un document, un vecteur global est calculé, en moyennant les vecteurs des mots on peut aussi utilisé des méthodes pondérées. Lorsqu’un utilisateur fournit une requête, celle-ci est également transformée en vecteur en suivant le même processus. Enfin, pour mesurer la similarité entre la requête et les textes, une métrique comme la similarité cosinus est utilisée : elle compare les vecteurs et attribue un score basé sur leur proximité dans l’espace vectoriel. Les résultats les plus proches sont ensuite renvoyés à l’utilisateur comme réponse pertinente à sa recherche.
Chatbot avec Word2Vec et TF-IDF: 
Dans mon chatbot, j’ai combiné TF-IDF et Word2Vec (en utilisant le modèle CBOW) pour exploiter à la fois l’importance statistique des mots et leur représentation sémantique. Voici en détail comment cela fonctionne :
Étape 1 : Préparation du corpus J’ai commencé par prétraiter le texte en supprimant les stopwords, en lemmatisant les mots et en ne gardant que les mots alphanumériques. Cela permet d’éliminer le bruit et de concentrer l’analyse sur les termes significatifs.
Étape 2 : TF-IDF pour l’importance des mots J’ai utilisé TF-IDF pour attribuer un poids à chaque mot en fonction de son importance dans le corpus.
La Fréquence Terme (TF) mesure combien de fois un mot apparaît dans une phrase par rapport au nombre total de mots dans cette phrase.
L’Inverse Document Frequency (IDF) mesure combien de phrases contiennent ce mot dans le corpus. Si un mot apparaît partout, son IDF est faible, sinon il est élevé.
En combinant ces deux mesures (TF * IDF), TF-IDF donne des poids plus élevés aux mots rares et significatifs. Chaque phrase est ainsi transformée en un vecteur de dimension égale au nombre de mots uniques dans le corpus.
Étape 3 : Word2Vec (CBOW) pour la sémantique J’ai formé un modèle Word2Vec en mode CBOW (Continuous Bag of Words). Dans ce mode, le modèle apprend à prédire un mot cible à partir des mots qui l’entourent dans une fenêtre contextuelle. Cela permet de capturer les relations sémantiques entre les mots. Par exemple, les mots « roi » et « reine » auront des vecteurs proches, car ils apparaissent dans des contextes similaires.

Étape 4 : Combiner TF-IDF et Word2Vec Une fois les modèles TF-IDF et Word2Vec générés, j’ai combiné leurs résultats pour créer une représentation enrichie des phrases :
Pour chaque mot d’une phrase, j’ai extrait son vecteur Word2Vec (représentation sémantique).
Ensuite, j’ai multiplié chaque vecteur Word2Vec par le poids TF-IDF du mot correspondant. Cette multiplication permet de pondérer les vecteurs Word2Vec en fonction de l’importance du mot dans la phrase. Par exemple, un mot rare et important aura un vecteur amplifié, tandis qu’un mot courant aura un vecteur réduit.
Étape 5 : Représentation des phrases Pour obtenir un vecteur unique représentant chaque phrase, j’ai calculé la moyenne des vecteurs pondérés par TF-IDF pour tous les mots de la phrase. Cela crée une représentation globale de la phrase dans l’espace vectoriel, tenant compte à la fois de l’importance des mots (TF-IDF) et de leurs relations sémantiques (Word2Vec).
Étape 6 : Calcul de la similarité Lorsqu’un utilisateur pose une question, je suis les mêmes étapes :
La question est prétraitée, vectorisée avec TF-IDF et enrichie avec les embeddings Word2Vec.
Une similarité cosinus est calculée entre le vecteur de la question et ceux des phrases du corpus. Les phrases les plus proches (en termes d’angle dans l’espace vectoriel) sont renvoyées comme réponses.
Cette combinaison TF-IDF + Word2Vec permet au chatbot de comprendre à la fois la pertinence statistique des mots et leurs relations sémantiques, offrant des résultats précis même pour des questions complexes ou reformulées.
J’ai également ajouté deux fonctionnalités importantes pour améliorer l’expérience utilisateur : des formules de politesse et la détection des questions hors contexte. Voici comment cela fonctionne :
Formules de politesse dans les réponses Lorsqu’une réponse est trouvée dans le corpus, j’ai implémenté un mécanisme pour personnaliser la réponse en fonction du type de question posée.
Si la question commence par « comment », la réponse est introduite par « Après analyse, ».
Pour les questions commençant par « pourquoi », la réponse est précédée de « Car, ».
Si l’utilisateur demande « peux-tu », le chatbot répond avec « Oui, bien sûr! » avant d’expliquer le résultat.
Ces formules de politesse rendent les interactions plus naturelles et engageantes, donnant l’impression d’une réponse contextualisée et adaptée.
Gestion des questions hors contexte Pour éviter de donner des réponses non pertinentes, j’ai ajouté un contrôle basé sur un seuil de similarité (par exemple, 0.2). Après avoir calculé la similarité cosinus entre la question de l’utilisateur et les phrases du corpus, si la meilleure correspondance a une similarité inférieure au seuil, le chatbot considère que la question est hors contexte. Dans ce cas, il informe l’utilisateur : « Il n'y a pas d'information sur ça. ».
Cette approche garantit que le chatbot ne renvoie pas des réponses inexactes ou dénuées de sens lorsque la question ne correspond pas aux données disponibles.
En résumé, en plus d’intégrer TF-IDF et Word2Vec pour une recherche précise, le chatbot est conçu pour offrir une interaction humaine en introduisant des formules de politesse et en évitant les réponses hors sujet, assurant ainsi une meilleure qualité d’expérience utilisateur.
Chatbot en utilisant un LLM via API call:
Pour ce chatbot, j’ai opté pour une approche totalement différente en utilisant l’API ChatGPT de OpenAI, basée sur le modèle GPT-4, un LLM prêt à l’emploi pour traiter efficacement les questions des utilisateurs. Voici comment cela fonctionne : lorsque l’utilisateur sélectionne le corpus où il souhaite effectuer sa recherche, le contenu textuel est extrait et stocké dans une variable appelée raw_texts. Ensuite, quand l’utilisateur pose sa question, celle-ci est enregistrée dans une variable question. Une fois ces étapes effectuées, une requête est envoyée à l’API du modèle GPT-4. On commence par définir un rôle pour le modèle, ce que l’on appelle un system prompt, et ici, c’est très simple : {"role": "system", "content": "You are a helpful assistant."}. Ensuite, une première requête est préparée, qui inclut à la fois le corpus choisi par l’utilisateur et la question posée. Cette requête ressemble à ceci : {"role": "user", "content": f"Based on the following text, answer the question or tell information that relate to the question respond in the correct language: '{question}'\n\nText: {raw_texts[0]}"}.
Le but est clair : demander au modèle de répondre précisément à la question de l’utilisateur, tout en respectant la langue dans laquelle la question a été posée, et en se basant uniquement sur le contenu du corpus fourni. Cette méthode est particulièrement efficace, car GPT-4, avec ses 1 000 milliards de paramètres, est entraîné pour comprendre des contextes complexes et générer des réponses pertinentes. Ce qui le rend encore plus adapté, c’est sa capacité à gérer jusqu’à 32 000 tokens de contexte, ce qui signifie que même des corpus très volumineux ne posent aucun problème. Le modèle est donc capable de fournir des réponses détaillées, précises et contextuellement appropriées, quelles que soient la taille ou la complexité des données textuelles analysées.

Implémentation du Nuage de Mots avec TF-IDF comme Spécifié sur Moodle:
Cette implémentation d’un générateur de nuage de mots interactif apporte une amélioration significative par rapport à une version basique en intégrant la possibilité de choisir entre un affichage classique ou basé sur les scores TF-IDF. Tandis qu’une approche classique se limite à représenter la fréquence brute des mots, la méthode TF-IDF permet de mettre en avant les termes les plus pertinents en fonction de leur importance relative dans le texte, offrant ainsi une visualisation plus significative dans des contextes où certains mots fréquents pourraient être peu informatifs. De plus, l’ajout d’options de personnalisation, comme la sélection de la couleur de fond, des palettes de couleurs ou du nombre maximal de mots, permet à l’utilisateur d’adapter la visualisation à ses besoins spécifiques, tout en garantissant une élimination avancée des mots vides grâce à une combinaison des listes de stopwords en anglais et en français.

Correction de la Recherche de Corpus dans d’autres Corpus:
Le but de cette partie de l'application est de calculer la similarité entre un corpus sélectionné par l'utilisateur et d'autres corpus, en fonction des choix effectués dans l'interface. Initialement, un problème est survenu, mais après analyse, le problème provenait de l'alignement des phrases extraites des fichiers sources et cibles, car le traitement ne garantissait pas une correspondance fiable entre les données brutes et leurs indices dans la matrice de similarité, ce qui causait une erreur. Pour résoudre cela, j'ai introduit un dictionnaire sentence_to_index, dont l'unique rôle est de permettre l'intégration cohérente des corpus dans les algorithmes de calcul. Cependant, il est essentiel de préciser que nous ne traitons pas ici de similarité entre phrases individuelles, mais bien de similarité entre des corpus entiers. Cette précision a permis de clarifier l'objectif principal de cette fonctionnalité et de garantir que le traitement des données s'aligne parfaitement avec les attentes de l'utilisateur.

Ajout des méthodes GloVe et Doc2Vec:
J'ai ajouté les modèles GloVe et Doc2Vec à l'implémentation existante. GloVe est un modèle de représentation vectorielle de mots basé sur des cooccurrences globales de mots dans un corpus, tandis que Doc2Vec est une extension de Word2Vec qui permet de représenter des documents entiers sous forme de vecteurs. Ces ajouts permettent d'élargir les options de vectorisation disponibles, offrant ainsi une plus grande flexibilité et précision dans le calcul des similarités entre phrases et documents.
