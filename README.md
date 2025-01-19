# Fonctionnalités

- **Recherche de texte (toutes combinaisons confondues)**
- **Streaming intelligent**
- **Calcul de similarité en utilisant les méthodes de vectorisation (Word2Vec, FastText)**
- **Implémentation de 2 chatbots** :
  - Chatbot avec Word2Vec et TF-IDF
  - Chatbot en utilisant un LLM via API call

# Améliorations et ajouts

- **Implémentation du Nuage de Mots avec TF-IDF comme Spécifié sur Moodle** :
  - Correction de la Recherche de Corpus dans d’autres Corpus.
  - Ajout des méthodes **GloVe** et **Doc2Vec**.

---

## Recherche de Texte (Toutes Combinaisons Confondues)

Grâce à l'application, on peut :
- Chercher un corpus dans d'autres corpus.
- Chercher un fichier (document) dans d'autres corpus.
- Chercher des phrases dans des fichiers ou corpus.
- Et bien d’autres combinaisons, simplement en spécifiant **‘quoi chercher’** et **‘où chercher’**.

L'application permet de :
- Décrire nos propres phrases ou textes.
- Uploader nos propres fichiers ou corpus pour une customisation maximale.
- Afficher les documents ou corpus les plus proches (**k plus proches**).  

Pour le calcul de similarité, plusieurs descripteurs et distances sont disponibles (tous ceux vus en cours). Ces fonctionnalités sont implémentées à l'aide de multiples fonctions présentes dans le code.

---

## Streaming Intelligent

Le système de **streaming intelligent** fonctionne en évaluant plusieurs méthodes de stemming pour sélectionner la plus adaptée.  

### Étapes détaillées :

1. **Entrées** :
   - `Q` : Le texte ou la phrase ("quoi chercher").
   - `C` : Le texte ou corpus ("où chercher").

2. **Lemmatisation** :
   - SpaCy est utilisé pour créer une version lemmatisée de `Q` et `C` :
     - `Q_lemma = lemmatize(Q)`
     - `C_lemma = lemmatize(C)`

3. **Stemming** :
   - Application de trois algorithmes de stemming sur `Q` :
     - `Q_stem_Porter = stem_Porter(Q)`
     - `Q_stem_Lancaster = stem_Lancaster(Q)`
     - `Q_stem_Snowball = stem_Snowball(Q)`

4. **Similarité TF-IDF** :
   - Calcul de la similarité entre chaque version stemmée et la version lemmatisée :
     ```
     similarity_S = (Q_lemma * Q_stem_S) / (|Q_lemma| * |Q_stem_S|)
     ```
     Où :
     - `*` : Produit scalaire des vecteurs TF-IDF.
     - `|x|` : Norme (longueur) du vecteur.

5. **Ratio de Réduction** :
   - Calcul du ratio de réduction pour chaque méthode :
     ```
     reduction_ratio_S = len(Q_stem_S) / len(Q)
     ```
   - Pénalité appliquée si le ratio est hors de l'intervalle `[0.5, 0.9]` :
     ```
     adjusted_similarity_S = similarity_S * 0.8 (si reduction_ratio_S < 0.5 ou > 0.9)
     adjusted_similarity_S = similarity_S (sinon)
     ```

6. **Comparaison avec le texte cible** :
   - Comparaison de chaque version stemmée avec `C_lemma` :
     ```
     context_similarity_S = (Q_stem_S * C_lemma) / (|Q_stem_S| * |C_lemma|)
     ```

7. **Score Final** :
   - Calcul du score final pour chaque méthode de stemming :
     ```
     final_score_S = 0.5 * adjusted_similarity_S + 0.5 * context_similarity_S
     ```

8. **Sélection de la Meilleure Méthode** :
   - La méthode avec le score final le plus élevé est choisie :
     ```
     best_stemmer = max(final_score_S)
     ```

---

## Calcul de Similarité en Utilisant les Méthodes de Vectorisation (Word2Vec, FastText)

Lorsqu’un utilisateur spécifie **quoi chercher** (mot/phrase) et **où chercher** (corpus/fichiers), le système utilise des méthodes de vectorisation comme **Word2Vec** ou **FastText** pour effectuer un calcul de similarité.

### Processus :
1. **Prétraitement** :
   - Nettoyage du texte (caractères spéciaux, mots vides, mise en minuscules).
   - Découpage en mots ou phrases.

2. **Vectorisation** :
   - Chaque mot/phrase est converti en vecteur à l'aide de **Word2Vec** ou **FastText**.

3. **Calcul du Vecteur Global** :
   - Moyenne des vecteurs des mots (avec pondération si nécessaire).

4. **Calcul de la Similarité** :
   - Une similarité cosinus est utilisée pour comparer les vecteurs.

Les résultats les plus proches sont renvoyés à l'utilisateur.

---

## Chatbot avec Word2Vec et TF-IDF

Ce chatbot combine **TF-IDF** et **Word2Vec (CBOW)** pour exploiter l’importance statistique des mots et leur représentation sémantique.  

### Étapes :
1. Prétraitement du corpus (stopwords, lemmatisation, etc.).
2. Utilisation de **TF-IDF** pour attribuer un poids à chaque mot.
3. Entraînement d'un modèle **Word2Vec CBOW** pour capturer les relations sémantiques.
4. Combinaison de **TF-IDF** et **Word2Vec** :
   - Multiplication du vecteur Word2Vec de chaque mot par son poids TF-IDF.
   - Moyennage des vecteurs pondérés pour obtenir un vecteur global.
5. Similarité Cosinus pour trouver les phrases les plus proches.

---

## Chatbot en Utilisant un LLM via API Call

Ce chatbot utilise l'API **ChatGPT** (modèle GPT-4) pour répondre aux questions des utilisateurs.  

### Processus :
1. Extraction du corpus choisi par l'utilisateur.
2. Conversion de la question en vecteur.
3. Envoi d'une requête à l'API :
   - Exemple de requête :
     ```json
     {
       "role": "user",
       "content": "Based on the following text, answer the question: '...'"
     }
     ```
4. Réponse contextuelle fournie par GPT-4.

Le modèle gère jusqu'à **32 000 tokens**, permettant d'analyser des corpus volumineux avec précision.

---

## Implémentation du Nuage de Mots avec TF-IDF

L'application intègre un générateur de nuage de mots interactif avec :
- Options d'affichage classique ou basé sur **TF-IDF**.
- Personnalisation (couleurs, nombre de mots, etc.).

---

## Ajout des Méthodes GloVe et Doc2Vec

Ajout des modèles **GloVe** et **Doc2Vec**, offrant :
- Représentation vectorielle des mots/documents.
- Options supplémentaires pour le calcul des similarités.

