
import os
import streamlit as st

st.set_page_config(layout="wide")





# import streamlit as st
# import os
# import streamlit as st
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import re

# # Fonction pour extraire le texte des dossiers
# import os
# import logging

# def chatbot(): 


#     # Configure logging to write errors to a log file instead of displaying them in Streamlit
#     logging.basicConfig(filename="errors.log", level=logging.ERROR)

#     def extract_text_from_folders(folders):
#         texts = []
#         for folder in folders:
#             for root, _, files in os.walk(folder):
#                 for file in files:
#                     if file.endswith(".txt"):  # Supposons que les fichiers sont des fichiers texte
#                         file_path = os.path.join(root, file)
#                         try:
#                             with open(file_path, "r", encoding="utf-8") as f:
#                                 texts.append(f.read())
#                         except Exception as e:
#                             # Log the error instead of showing it in Streamlit
#                             logging.error(f"Erreur lors de la lecture du fichier {file_path}: {e}")
#         return texts


#     # Prétraitement du texte : Tokenisation et nettoyage
#     def preprocess_text(text):
#         sentences = re.split(r'(?<=[.!?])\s+', text)  # Découper en phrases
#         cleaned_sentences = [re.sub(r'\W+', ' ', sentence.lower()).strip() for sentence in sentences]
#         return cleaned_sentences

#     # Calculer la similarité entre une question et les phrases du corpus
#     def find_most_relevant_sentence(question, corpus_sentences):
#         vectorizer = TfidfVectorizer()
#         vectors = vectorizer.fit_transform(corpus_sentences + [question])
#         similarity_scores = cosine_similarity(vectors[-1], vectors[:-1]).flatten()
#         max_index = similarity_scores.argmax()
#         return corpus_sentences[max_index], similarity_scores[max_index]

#     # Détection du mot le plus significatif
#     def detect_key_word(question, vectorizer):
#         tfidf_vector = vectorizer.transform([question])
#         max_tfidf_index = tfidf_vector.toarray().argmax()
#         return vectorizer.get_feature_names_out()[max_tfidf_index]

#     # Streamlit app
#     def main():
#         st.title("Chatbot avec Recherche dans un Corpus")
        
#         # Définir les dossiers du corpus
#         MAIN_FOLDERS = [
#             "/Users/yanis/Desktop/yahiaoui-app/Text-Analytics-Tool/Corpus_Anglais",
#             "/Users/yanis/Desktop/yahiaoui-app/Text-Analytics-Tool/Corpus_Francais"
#         ]

#         # Extraction de texte
#         st.header("Chargement du corpus...")
#         with st.spinner("Extraction du texte des dossiers..."):
#             raw_texts = extract_text_from_folders(MAIN_FOLDERS)
#             if not raw_texts:
#                 st.error("Aucun texte trouvé dans les dossiers spécifiés.")
#                 return

#         # Prétraitement du corpus
#         st.header("Prétraitement du corpus...")
#         corpus_sentences = []
#         for text in raw_texts:
#             corpus_sentences.extend(preprocess_text(text))

#         st.success(f"Corpus chargé avec {len(corpus_sentences)} phrases.")
        
#         # Entrée de la question par l'utilisateur
#         question = st.text_input("Posez une question :")
#         if question:
#             # Prétraiter la question
#             cleaned_question = preprocess_text(question)[0]

#             # Recherche de la réponse
#             relevant_sentence, similarity_score = find_most_relevant_sentence(cleaned_question, corpus_sentences)

#             # Identification du mot-clé le plus significatif
#             vectorizer = TfidfVectorizer()
#             vectorizer.fit(corpus_sentences + [cleaned_question])
#             key_word = detect_key_word(cleaned_question, vectorizer)

#             # Afficher la réponse
#             st.markdown("### Résultat")
#             st.write(f"**Phrase la plus pertinente :** {relevant_sentence}")
#             st.write(f"**Score de similarité :** {similarity_score:.4f}")
#             st.write(f"**Mot-clé détecté :** {key_word}")

#             # Ajouter une formule de politesse
#             polite_response = ""
#             if "comment" in question.lower():
#                 polite_response = "Après analyse, "
#             elif "pourquoi" in question.lower():
#                 polite_response = "Car, "
#             elif "peux-tu" in question.lower():
#                 polite_response = "Oui, bien sûr! "

#             st.write(f"**Réponse complète :** {polite_response}{relevant_sentence}")

#     # Exécuter l'application Streamlit
#     if __name__ == "__main__":
#         main()





# def chatbot():
#     import streamlit as st
#     import os
#     import logging
#     import spacy
#     import numpy as np
#     from sklearn.feature_extraction.text import TfidfVectorizer
#     from sklearn.metrics.pairwise import cosine_similarity

#     # Charger le modèle linguistique spaCy
#     nlp = spacy.load("fr_core_news_sm")

#     # Configurer les logs pour gérer les erreurs
#     logging.basicConfig(filename="errors.log", level=logging.ERROR)

#     # Fonction pour extraire le texte des dossiers
#     def extract_text_from_folders(folders):
#         texts = []
#         for folder in folders:
#             for root, _, files in os.walk(folder):
#                 for file in files:
#                     if file.endswith(".txt"):
#                         file_path = os.path.join(root, file)
#                         try:
#                             with open(file_path, "r", encoding="utf-8") as f:
#                                 texts.append(f.read())
#                         except Exception as e:
#                             logging.error(f"Erreur lors de la lecture du fichier {file_path}: {e}")
#         return texts

#     # Prétraitement du texte avec spaCy (version pour similarité)
#     def preprocess_text_spacy(text):
#         doc = nlp(text)
#         cleaned_sentences = [
#             " ".join(token.lemma_ for token in sent if token.is_alpha and not token.is_stop)
#             for sent in doc.sents
#         ]
#         return cleaned_sentences

#     # Mapping entre phrases originales et prétraitées
#     def create_mapping(original_texts):
#         mapping = {}
#         for text in original_texts:
#             doc = nlp(text)
#             for sent in doc.sents:
#                 original_sentence = sent.text.strip()
#                 preprocessed_sentence = " ".join(
#                     token.lemma_ for token in sent if token.is_alpha and not token.is_stop
#                 )
#                 mapping[preprocessed_sentence] = original_sentence
#         return mapping

#     # Calculer la similarité entre une question et les phrases du corpus
#     def compute_similarity(question, corpus_sentences):
#         vectorizer = TfidfVectorizer()
#         vectors = vectorizer.fit_transform(corpus_sentences + [question])
#         similarity_scores = cosine_similarity(vectors[-1], vectors[:-1]).flatten()
#         return similarity_scores, vectorizer

#     # Identifier les mots-clés significatifs
#     def detect_key_words(question, vectorizer, top_n=3):
#         tfidf_vector = vectorizer.transform([question])
#         tfidf_scores = tfidf_vector.toarray()[0]
#         top_indices = np.argsort(tfidf_scores)[::-1][:top_n]
#         return [vectorizer.get_feature_names_out()[index] for index in top_indices]

#     # Application Streamlit
#     def main():
#         st.title("Chatbot avec Recherche dans un Corpus")

#         # Définir les dossiers du corpus
#         MAIN_FOLDERS = [
#             "/Users/yanis/Desktop/yahiaoui-app/Text-Analytics-Tool/Corpus_Anglais",
#             "/Users/yanis/Desktop/yahiaoui-app/Text-Analytics-Tool/Corpus_Francais",
#         ]

#         # Extraction de texte
#         st.header("Chargement du corpus...")
#         with st.spinner("Extraction du texte des dossiers..."):
#             raw_texts = extract_text_from_folders(MAIN_FOLDERS)
#             if not raw_texts:
#                 st.error("Aucun texte trouvé dans les dossiers spécifiés.")
#                 return

#         # Création du mapping entre phrases originales et prétraitées
#         st.header("Prétraitement du corpus...")
#         mapping = create_mapping(raw_texts)
#         preprocessed_sentences = list(mapping.keys())
#         st.success(f"Corpus chargé avec {len(preprocessed_sentences)} phrases.")

#         # Entrée de la question par l'utilisateur
#         question = st.text_input("Posez une question :")
#         if question:
#             # Prétraiter la question
#             preprocessed_question = preprocess_text_spacy(question)[0]

#             # Calculer les similarités
#             similarity_scores, vectorizer = compute_similarity(preprocessed_question, preprocessed_sentences)

#             # Extraire les mots-clés significatifs
#             key_words = detect_key_words(preprocessed_question, vectorizer)

#             # Trier les phrases par similarité décroissante
#             ranked_sentences = sorted(
#                 zip(preprocessed_sentences, similarity_scores), key=lambda x: x[1], reverse=True
#             )

#             # Ajouter une formule de politesse
#             polite_response = ""
#             if "comment" in question.lower():
#                 polite_response = "Après analyse, "
#             elif "pourquoi" in question.lower():
#                 polite_response = "Car, "
#             elif "peux-tu" in question.lower():
#                 polite_response = "Oui, bien sûr! "

#             # Afficher les résultats
#             st.markdown("### Résultats")
#             st.write(f"**Mots-clés détectés :** {', '.join(key_words)}")

#             st.write("**Phrases classées par similarité :**")
#             for idx, (preprocessed_sentence, score) in enumerate(ranked_sentences[:5], 1):
#                 original_sentence = mapping[preprocessed_sentence]
#                 st.write(f"{idx}. **Phrase :** {original_sentence} (Score : {score:.4f})")

#             # Réponse finale
#             best_sentence = mapping[ranked_sentences[0][0]]
#             st.write(f"**Réponse complète :** {polite_response}{best_sentence}")

#     # Exécuter l'application Streamlit
#     if __name__ == "__main__":
#         main()




def chatbot():
    import streamlit as st
    import os
    import logging
    import spacy
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    # Charger le modèle linguistique spaCy
    nlp = spacy.load("fr_core_news_sm")

    # Configurer les logs pour gérer les erreurs
    logging.basicConfig(filename="errors.log", level=logging.ERROR)

    # Fonction pour extraire le texte des dossiers
    import os
    import logging

    import os
    import logging

    def extract_text_from_folders(folders):
        texts = []
        for folder in folders:
            if isinstance(folder, str) and os.path.isfile(folder):  # If it's a file, read it directly
                if folder.endswith(".txt"):
                    try:
                        with open(folder, "r", encoding="utf-8") as f:
                            texts.append(f.read())
                    except Exception as e:
                        logging.error(f"Erreur lors de la lecture du fichier {folder}: {e}")
            elif isinstance(folder, str) and os.path.isdir(folder):  # If it's a directory, walk through it
                for root, _, files in os.walk(folder):
                    for file in files:
                        if file.endswith(".txt"):
                            file_path = os.path.join(root, file)
                            try:
                                with open(file_path, "r", encoding="utf-8") as f:
                                    texts.append(f.read())
                            except Exception as e:
                                logging.error(f"Erreur lors de la lecture du fichier {file_path}: {e}")
            elif isinstance(folder, str) and folder.endswith(".txt"):  # Direct file content handling
                texts.append(folder)  # This assumes the folder is actually a string of file content
            else:
                logging.warning(f"{folder} n'est ni un fichier texte ni un dossier valide.")
        return texts


    # Prétraitement du texte avec spaCy (version pour similarité)
    def preprocess_text_spacy(text):
        doc = nlp(text)
        cleaned_sentences = [
            " ".join(token.lemma_ for token in sent if token.is_alpha and not token.is_stop)
            for sent in doc.sents
        ]
        return cleaned_sentences

    # Mapping entre phrases originales et prétraitées
    def create_mapping(original_texts):
        mapping = {}
        for text in original_texts:
            doc = nlp(text)
            for sent in doc.sents:
                original_sentence = sent.text.strip()
                preprocessed_sentence = " ".join(
                    token.lemma_ for token in sent if token.is_alpha and not token.is_stop
                )
                mapping[preprocessed_sentence] = original_sentence
        return mapping

    # Calculer la similarité entre une question et les phrases du corpus
    def compute_similarity(question, corpus_sentences):
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(corpus_sentences + [question])
        similarity_scores = cosine_similarity(vectors[-1], vectors[:-1]).flatten()
        return similarity_scores, vectorizer

    # Identifier les mots-clés significatifs
    def detect_key_words(question, vectorizer, top_n=3):
        tfidf_vector = vectorizer.transform([question])
        tfidf_scores = tfidf_vector.toarray()[0]
        top_indices = np.argsort(tfidf_scores)[::-1][:top_n]
        return [vectorizer.get_feature_names_out()[index] for index in top_indices]

    # Application Streamlit
    def main():
        st.title("Chatbot avec Recherche dans un Corpus")

        # Définir les dossiers du corpus
        CORPUS_OPTIONS = {
            "Corpus Anglais": "/Users/yanis/Desktop/yahiaoui-app/Text-Analytics-Tool/Corpus_Anglais",
            "Corpus Français": "/Users/yanis/Desktop/yahiaoui-app/Text-Analytics-Tool/Corpus_Francais",
        }

        # Choisir un corpus ou importer un fichier/dossier
        st.header("Choisissez une option pour la recherche")
        selected_option = st.selectbox(
            "Sélectionnez une option :",
            ["Corpus Anglais", "Corpus Français", "Importer un fichier ou dossier"]
        )

        if selected_option in CORPUS_OPTIONS:
            selected_folder = CORPUS_OPTIONS[selected_option]
            folders = [selected_folder]
            p = 1
        else:
            uploaded_file = st.file_uploader("Importez un fichier texte", type=["txt"])
            if uploaded_file is not None:
                p = 2
                if uploaded_file.name.endswith(".txt"):
                    # If a .txt file is uploaded, read its content directly
                    try:
                        file_content = uploaded_file.getvalue().decode("utf-8")
                        st.write("Contenu du fichier chargé avec succès :")
                        st.text_area("Contenu du fichier", file_content, height=200)

                        # Create the structure as a list with the entire text as the first item
                        text_structure = [file_content.strip()]  # Store the whole content as the first item in the list

                        # Display the resulting structure
                        st.write("Structure du texte :")
                        st.write(text_structure)

                    except Exception as e:
                        st.error(f"Erreur lors de la lecture du fichier : {e}")
                else:
                    st.error("Veuillez importer un fichier texte (.txt).")

        # Extraction de texte
        st.header("Chargement du corpus...")
        if p == 1:
            with st.spinner("Extraction du texte..."):
                raw_texts = extract_text_from_folders(folders)
                
                if not raw_texts:
                    st.error("Aucun texte trouvé dans le dossier ou fichier sélectionné.")
                    return
        else:
            raw_texts = [file_content]
            

        # Création du mapping entre phrases originales et prétraitées
        st.header("Prétraitement du corpus...")
        mapping = create_mapping(raw_texts)
        preprocessed_sentences = list(mapping.keys())
        st.success(f"Corpus chargé avec {len(preprocessed_sentences)} phrases.")

            # Entrée de la question par l'utilisateur
    # Entrée de la question par l'utilisateur
        question = st.text_input("Posez une question :")
        if question:
            # Prétraiter la question
            preprocessed_question = preprocess_text_spacy(question)[0]

            # Calculer les similarités
            similarity_scores, vectorizer = compute_similarity(preprocessed_question, preprocessed_sentences)

            # Extraire les mots-clés significatifs
            key_words = detect_key_words(preprocessed_question, vectorizer)

            # Trier les phrases par similarité décroissante
            ranked_sentences = sorted(
                zip(preprocessed_sentences, similarity_scores), key=lambda x: x[1], reverse=True
            )

            # Définir un seuil minimum de similarité (par exemple : 0.2)
            similarity_threshold = 0.2

            # Vérifier si la meilleure correspondance dépasse le seuil
            if ranked_sentences[0][1] < similarity_threshold:
                st.markdown("### Résultats")
                st.write("**Il n'y a pas d'information sur ça.**")
            else:
                # Ajouter une formule de politesse
                polite_response = ""
                if "comment" in question.lower():
                    polite_response = "Après analyse, "
                elif "pourquoi" in question.lower():    
                    polite_response = "Car, "
                elif "peux-tu" in question.lower():
                    polite_response = "Oui, bien sûr! "

                # Afficher les résultats
                st.markdown("### Résultats")
                st.write(f"**Mots-clés détectés :** {', '.join(key_words)}")

                st.write("**Phrases classées par similarité :**")
                for idx, (preprocessed_sentence, score) in enumerate(ranked_sentences[:5], 1):
                    original_sentence = mapping[preprocessed_sentence]
                    st.write(f"{idx}. **Phrase :** {original_sentence} (Score : {score:.4f})")

                # Réponse finale
                best_sentence = mapping[ranked_sentences[0][0]]
                st.write(f"**Réponse complète :** {polite_response}{best_sentence}")

    # Exécuter l'application Streamlit
    if __name__ == "__main__":
        main()




# Les textes du corpus sont divisés en phrases grâce au modèle linguistique spaCy.
# Chaque phrase est nettoyée pour ne conserver que les lemmes (formes de base des mots)
# et éliminer les mots non pertinents comme les stop words ou les symboles.
# Un mapping est créé pour lier chaque phrase nettoyée à sa phrase originale.
# tf-idf utilisée pour représenter les phrases du corpus et la question comme des vecteurs.
# Les similarités entre la question et les phrases du corpus sont calculées avec la mesure cosine_similarity.
# Les mots les plus significatifs dans la question sont identifiés grâce aux scores TF-IDF.
# 5 phrases les plus pertinentes sont affichées avec leurs scores de similarité.
# Une réponse polie et adaptée au type de question (par exemple, "comment", "pourquoi", etc.) est ajoutée avant de présenter la phrase la plus pertinente.




import streamlit as st
from corpus_to_corpus import corpus_to_corpus
from distances import distances
from document_to_document import document_to_document
from vectorisation import vectorisation
from llmchatbot import llmchatbot   
# Définir les pages disponibles et leurs fonctions
pages = {
    "📚 Chercher les corpus les plus proches": corpus_to_corpus,
    "📏 Comparer selon les distances": distances,
    "📄 Chercher les documents les plus proches": document_to_document,
    "🤖 Chatbot": chatbot,
    "🤖 llm Chatbot": llmchatbot,
    "🧠 Calculer la similarité avec vectorisation": vectorisation,
}

# Fonction pour changer la langue
def change_language():
    language = st.sidebar.selectbox("Choisissez la langue", ["Français", "English"])
    if language == "Français":
        st.session_state["language"] = "fr"
    else:
        st.session_state["language"] = "en"

# Initialiser la page actuelle
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "Home"

# Initialiser la langue (par défaut Français)
if "language" not in st.session_state:
    st.session_state["language"] = "fr"

# Change la langue


# Traductions en fonction de la langue
translations = {
    "fr": {
        "title": "🧪 Outil d'Analyse de Similarité de Textes",
        "navigation_title": "🔍 Navigation",
        "tip": "💡 **Astuce :** Explorez les différentes sections pour découvrir les capacités de l'application !",
        "choose_function": "Choisissez une fonctionnalité :",
        "logo_caption": "TextAnalizer"
    },
    "en": {
        "title": "🧪 Text Similarity Analysis Tool",
        "navigation_title": "🔍 Navigation",
        "tip": "💡 **Tip:** Explore the different sections to discover the capabilities of the app!",
        "choose_function": "Choose a functionality:",
        "logo_caption": "TextAnalizer"
    }
}

# Barre latérale de navigation
with st.sidebar:
    # Ajout du logo avec une taille réduite
    st.image(
        "/Users/yanis/Desktop/yahiaoui-app/Codium Ai Icon.png", 
        width=150 , caption=translations[st.session_state["language"]]["logo_caption"]
    )
    change_language()
    st.title(translations[st.session_state["language"]]["navigation_title"])
    st.write("Utilisez les options ci-dessous pour naviguer entre les différentes fonctionnalités.")
    
    # Navigation avec radio buttons
    st.session_state["current_page"] = st.radio(
        translations[st.session_state["language"]]["choose_function"], 
        list(pages.keys())
    )

    st.markdown("---")
    st.write(translations[st.session_state["language"]]["tip"])

# En-tête principal
st.title(translations[st.session_state["language"]]["title"])

# Rendre la page sélectionnée
page_function = pages[st.session_state["current_page"]]
page_function()

