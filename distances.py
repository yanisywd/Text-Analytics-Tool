
import streamlit as st

def distances():
    import streamlit as st

    # Define the different pages



    import os
    import streamlit as st
    

    # Function to recursively list files and folders
    def list_arborescence(path):
        arborescence = {}
        for root, dirs, files in os.walk(path):
            # Exclude `.DS_Store` from files
            files = [f for f in files if f != ".DS_Store"]
            relative_root = os.path.relpath(root, path)
            arborescence[relative_root] = {
                "folders": dirs,
                "files": files
            }
        return arborescence

    # Function to read the content of a file
    def read_file_content(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                return file.read()
        except Exception as e:
            return f"Error reading {file_path}: {e}"

    # Function to split text into sentences
    def split_into_sentences(text):
        # Simple sentence splitting based on punctuation. You can improve this.
        sentences = text.split('. ')
        return [s.strip() + '.' for s in sentences if s.strip()]

    # Recursive function to collect text from all files within a directory
    def collect_text_from_directory(base_path, relative_path, selected_items):
        full_path = os.path.join(base_path, relative_path)
        for root, dirs, files in os.walk(full_path):
            # Exclude `.DS_Store` from files
            files = [f for f in files if f != ".DS_Store"]
            for file in files:
                file_path = os.path.join(root, file)
                # Read file content
                content = read_file_content(file_path)
                # Split content into sentences
                sentences = split_into_sentences(content)
                # Create a unique variable name for each file
                variable_name = f"file_{os.path.relpath(file_path, base_path).replace('/', '_').replace(' ', '_')}"
                selected_items[variable_name] = {
                    "file_content": content,
                    "sentences": sentences
                }

    # Function to handle user selection in an arborescence
    def handle_selection(tree, base_paths, side):
        selected_items = {}
        st.write(f"### {side}")
        
        for folder, contents in tree.items():
            # Folder-level selection
            folder_selected = st.checkbox(f"📂 {folder}", key=f"{side}_folder_{folder}")
            
            if folder_selected:
                # Collect text from all files within the selected folder
                for base_path in base_paths:
                    full_path = os.path.join(base_path, folder)
                    if os.path.exists(full_path):
                        collect_text_from_directory(base_path, folder, selected_items)
                        break
            
            # File-level selection with expandable view
            with st.expander(f"📂 {folder}"):
                for file in contents["files"]:
                    # Try to find the full file path by checking all base paths
                    file_path = None
                    for base_path in base_paths:
                        potential_path = os.path.join(base_path, folder, file)
                        if os.path.exists(potential_path):
                            file_path = potential_path
                            break
                    
                    if not file_path:
                        st.warning(f"Could not find file: {file}")
                        continue
                    
                    # Checkbox for entire file
                    file_selected = st.checkbox(f"📄 {file}", key=f"{side}_file_{file_path}")
                    
                    if file_selected:
                        # Read file content
                        file_content = read_file_content(file_path)
                        sentences = split_into_sentences(file_content)
                        
                        # Sentence-level selection
                        st.write("Select specific sentences:")
                        file_sentences_selected = False
                        
                        for i, sentence in enumerate(sentences):
                            sentence_selected = st.checkbox(
                                f"Sentence {i+1}: {sentence}", 
                                key=f"{side}_sentence_{file_path}_{i}"
                            )
                            
                            if sentence_selected:
                                file_sentences_selected = True
                                # Create a unique variable name for the selected sentence
                                variable_name = f"file_{file.replace('/', '_').replace(' ', '_')}_sentence_{i}"
                                selected_items[variable_name] = {
                                    "file_content": sentence,  # Only the selected sentence
                                    "sentences": [sentence],
                                    "original_file": file_path,
                                    "sentence_number": i
                                }
                        
                        # If no sentences selected, add the whole file
                        if not file_sentences_selected:
                            variable_name = f"file_{file.replace('/', '_').replace(' ', '_')}"
                            selected_items[variable_name] = {
                                "file_content": file_content,
                                "sentences": sentences,
                                "original_file": file_path
                            }
        
        return selected_items
    # Main function for the Streamlit app











    import re
    import numpy as np
    import pandas as pd
    from scipy.spatial.distance import pdist, squareform



    # 1. Diviser le texte en phrases
    def split_into_sentences(text):
        # Utilisation d'une expression régulière pour diviser les phrases sur les points, points d'exclamation, points d'interrogation
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return sentences

    # 2. Prétraitement du texte
    def preprocess_text(sentences):
        unique_tokens = set()  # Utiliser un ensemble pour les mots uniques
        for sentence in sentences:
            # Nettoyer et tokeniser
            tokens = re.findall(r'\b\w+\b', sentence.lower())
            unique_tokens.update(tokens)  # Ajouter les mots uniques
        return sorted(unique_tokens)  # Retourner une liste triée

    # 3. Créer la matrice binaire et la matrice d'occurrences
    def create_matrices(sentences, unique_tokens):
        binary_matrix = []
        occurrence_matrix = []
        
        for sentence in sentences:
            # Tokeniser la phrase
            tokens = re.findall(r'\b\w+\b', sentence.lower())
            
            # Créer une ligne pour la matrice binaire
            binary_row = [1 if token in tokens else 0 for token in unique_tokens]
            binary_matrix.append(binary_row)
            
            # Créer une ligne pour la matrice d'occurrences
            occurrence_row = [tokens.count(token) for token in unique_tokens]
            occurrence_matrix.append(occurrence_row)

        return np.array(binary_matrix), np.array(occurrence_matrix)

    # 4. Calculer la distance de Manhattan
    def calculate_manhattan_distance(matrix):
        return squareform(pdist(matrix, metric='cityblock'))

    # 5. Calculer la distance euclidienne
    def calculate_euclidean_distance(matrix):
        return squareform(pdist(matrix, metric='euclidean'))


    def calculate_jaccard_distance(matrix):
        return squareform(pdist(matrix, metric='jaccard'))

    def calculate_hamming_distance(matrix):
        return squareform(pdist(matrix, metric='hamming'))

    def calculate_bray_curtis_distance(matrix):
        return squareform(pdist(matrix, metric='braycurtis'))

    from scipy.spatial.distance import pdist, squareform

    def calculate_cosine_distance(matrix):
        return squareform(pdist(matrix, metric='cosine'))




    from scipy.special import rel_entr
    import numpy as np

    import numpy as np
    from scipy.special import rel_entr

    def calculate_kullback_leibler_distance(matrix):
        num_docs = matrix.shape[0]
        kl_matrix = np.zeros((num_docs, num_docs))

        # Small epsilon to avoid zero probabilities
        epsilon = 1e-10

        for i in range(num_docs):
            # Normalize the distribution for the document i
            p = matrix[i] + epsilon  # Add epsilon to avoid zero
            p /= np.sum(p)  # Normalize to sum to 1

            for j in range(num_docs):
                # Normalize the distribution for the document j
                q = matrix[j] + epsilon  # Add epsilon to avoid zero
                q /= np.sum(q)  # Normalize to sum to 1
                
                # Calculate KL divergence
                kl_divergence = np.sum(rel_entr(p, q))
                kl_matrix[i, j] = kl_divergence

        return kl_matrix



    def calculate_similarity_matrix(distance_matrix):
        max_distance = np.max(distance_matrix)
        return 1 - (distance_matrix / max_distance)
    # Calculer les matrices de similarité

    def k_plus_proches_documents(doc_requete_index, similarity_matrix, k):
        """
        Find the k nearest documents based on the similarity matrix.
        
        Parameters:
        - doc_requete_index: Index of the document to query (0-indexed).
        - similarity_matrix: Precomputed similarity matrix.
        - k: Number of nearest neighbors to return.
        
        Returns:
        - List of tuples (index, similarity_score) of the k nearest documents.
        """
        # Get similarity scores for the query document
        similarity_scores = similarity_matrix[doc_requete_index]

        # Get indices of the top k similar documents, excluding the query document itself
        top_k_indices = np.argsort(similarity_scores)[-k-1:-1][::-1]

        # Get the top k documents and their similarity scores
        top_k_documents = [(i, similarity_scores[i]) for i in top_k_indices]
        
        return top_k_documents



    def normalize_matrix(matrix):
        """Normalise chaque ligne de la matrice en probabilités (fréquences relatives)."""
        row_sums = matrix.sum(axis=1, keepdims=True)
        normalized_matrix = matrix / row_sums
        return normalized_matrix


    def normalize_matrix_l1(matrix):
        """
        Normalise chaque ligne de la matrice avec la norme L1 (somme des éléments de la ligne = 1).
        """
        row_sums = matrix.sum(axis=1, keepdims=True)
        normalized_matrix = matrix / row_sums
        return normalized_matrix

    def normalize_matrix_l2(matrix):
        """
        Normalise chaque ligne de la matrice avec la norme L2 (racine carrée de la somme des carrés des éléments).
        """
        row_sums = np.linalg.norm(matrix, axis=1, keepdims=True)
        normalized_matrix = matrix / row_sums
        return normalized_matrix





    def k_plus_proches_documents_custom_query(query, documents, k):
        """
        Finds the K nearest documents to a custom query using a similarity metric.

        Parameters:
        - query: The custom query string.
        - documents: List of all documents in the corpus.
        - k: Number of nearest documents to return.

        Returns:
        - List of tuples (index, similarity_score) of the K nearest documents.
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        # Combine the query and corpus into one list for vectorization
        all_texts = [query] + documents

        # Calculate the TF-IDF matrix
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(all_texts)

        # Calculate cosine similarity between the query (first row) and the corpus (remaining rows)
        similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

        # Get the top K similar documents
        top_k_indices = np.argsort(similarity_scores)[-k:][::-1]
        nearest_docs = [(i, similarity_scores[i]) for i in top_k_indices]

        return nearest_docs


    def create_tfidf_matrices(sentences, unique_tokens, binary_matrix, occurrence_matrix):
        N = len(sentences)  # Nombre de phrases
        
        # Calcul de l'IDF directement dans la même fonction
        idf_values = []
        for token in unique_tokens:
            containing_sentences = sum(1 for sentence in sentences if token in re.findall(r'\b\w+\b', sentence.lower()))
            idf = np.log((N + 1) / (containing_sentences + 1)) + 1
            idf_values.append(idf)
        
        idf_values = np.array(idf_values)
        
        # Calcul des matrices TF-IDF
        tfidf_binary_matrix = binary_matrix * idf_values
        tfidf_occurrence_matrix = occurrence_matrix * idf_values
        
        # TF-IDF Probabilité (normalisation des occurrences)
        tf_prob_matrix = occurrence_matrix / np.sum(occurrence_matrix, axis=1, keepdims=True)
        tfidf_prob_matrix = tf_prob_matrix * idf_values
        

        return tfidf_binary_matrix, tfidf_occurrence_matrix, tfidf_prob_matrix




    st.title("Calcule la similarite et chercher des documents :")

    # Define main folders
    MAIN_FOLDERS = [
        "/Users/yanis/Desktop/yahiaoui-app/Text-Analytics-Tool/Corpus_Anglais",
        "/Users/yanis/Desktop/yahiaoui-app/Text-Analytics-Tool/Corpus_Francais"
    ]

    # Load arborescences for both corpora
    corpus_anglais = list_arborescence(MAIN_FOLDERS[0])
    corpus_francais = list_arborescence(MAIN_FOLDERS[1])

    # Combine corpora
    combined_corpus = {**corpus_anglais, **corpus_francais}

    # Layout: Two columns
    col1, col2 = st.columns(2)

    # Left column: "Quoi chercher ?"
    with col1:
        st.header("Quoi chercher ?")
        st.markdown("---")
        st.write("** sélectionnez un fichier source :**")
        st.button("📂 Sélectionner un fichier source")  # Fake button for UI purposes
        manual_source_sentence = st.text_area("Ou Enter source sentence manually:", "")
        if manual_source_sentence:
            selected_text_source = {"manual_input": {"file_content": manual_source_sentence, "sentences": [manual_source_sentence]}}
        else:
            selected_text_source = handle_selection(combined_corpus, MAIN_FOLDERS, "Quoi chercher")
        
        # Placeholder for file selection (Source)


    with col2:
        st.header("Où chercher ?")
        st.markdown("---")
        st.write("** sélectionnez un fichier cible :**")
        st.button("📂 Sélectionner un fichier cible")  # Fake button for UI purposes
        manual_target_sentence = st.text_area("Ou Enter target sentence manually:", "")
        if manual_target_sentence:
            selected_text_target = {"manual_input": {"file_content": manual_target_sentence, "sentences": [manual_target_sentence]}}
        else:
            selected_text_target = handle_selection(combined_corpus, MAIN_FOLDERS, "Où chercher ?")
        
        # Placeholder for file selection (Target)




    # Display variables for debugging
    st.write("## Selected Variables (Source)")
    for var_name, content in selected_text_source.items():
        st.write(f"**{var_name}:** {content['file_content'][:100]}...")  # Display the first 100 characters for preview

    st.write("## Selected Variables (Target)")
    for var_name, content in selected_text_target.items():
        st.write(f"**{var_name}:** {content['file_content'][:100]}...")  # Display the first 100 characters for preview









    selected_descripteur = st.sidebar.selectbox('Descripteur', ('Binaire','Occurence','Binaire normalise L1','Binaire normalise L2','Occurence Normalisé L1', 'Occurence Normalisé L2', 'TF-IDF Binaire', 
    'TF-IDF Occurence', 'TF-IDF Probabilité'))

    selected_distance = st.sidebar.selectbox('Distance', ('Euclidienne', 'Manhattan', 'Cosinus','Curtis','Leibler','Jaccard','Hamming'))





    # if True:
    #     sentences = source_content + target_content
    #     st.write(f"les sentences : {sentences}")
    #     tokens = preprocess_text(sentences)
    #     token_count = len(tokens)
    #     sentences_no_punctuation_count = len(sentences)

    # print(sentences_no_punctuation_count , token_count)



    # for source_var, source_content in selected_text_source.items():
    #     st.write(f" this is the source content : {source_content["file_content"]}")



    # Create an empty list to store sentences




    # Comparison logic for files and sentences

    import streamlit as st
    import spacy
    from nltk.stem import PorterStemmer

    # Define the path to the stopwords file
    STOPWORDS_FILE = "/Users/yanis/Desktop/yahiaoui-app/Text-Analytics-Tool/stopwords.txt"

    # Function to load stopwords from a file
    def load_stopwords():
        try:
            with open(STOPWORDS_FILE, "r", encoding="utf-8") as f:
                stopwords = f.read().splitlines()
            return set(stopwords)
        except Exception as e:
            st.error(f"Error loading stopwords: {e}")
            return set()

    # Function to remove stopwords from a single sentence
    def remove_stopwords(sentence, stopwords):
        words = sentence.split()
        filtered_words = [word for word in words if word.lower() not in stopwords]
        return " ".join(filtered_words)

    # Initialize spaCy and NLTK
    nlp = spacy.load("en_core_web_sm")
    stemmer = PorterStemmer()

    # Lemmatization function (using spaCy)
    def lemmatize_sentence(sentence):
        doc = nlp(sentence)
        lemmatized_sentence = " ".join([token.lemma_ for token in doc])
        return lemmatized_sentence

    # Stemming function (using NLTK)
    def stem_sentence(sentence):
        words = sentence.split()
        stemmed_sentence = " ".join([stemmer.stem(word) for word in words])
        return stemmed_sentence



    # import spacy
    # from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer
    # from nltk.corpus import wordnet
    # from sklearn.feature_extraction.text import TfidfVectorizer

    # # Load Spacy English model
    # nlp = spacy.load("en_core_web_sm")

    # # Initialize stemmers
    # porter_stemmer = PorterStemmer()
    # lancaster_stemmer = LancasterStemmer()
    # snowball_stemmer = SnowballStemmer("english")

    # # Lemmatizer function
    # def lemmatize_sentence(sentence):
    #     doc = nlp(sentence)
    #     lemmatized_sentence = " ".join([token.lemma_ for token in doc])
    #     return lemmatized_sentence

    # # Stemming functions
    # def porter_stem(sentence):
    #     words = sentence.split()
    #     return " ".join([porter_stemmer.stem(word) for word in words])

    # def lancaster_stem(sentence):
    #     words = sentence.split()
    #     return " ".join([lancaster_stemmer.stem(word) for word in words])

    # def snowball_stem(sentence):
    #     words = sentence.split()
    #     return " ".join([snowball_stemmer.stem(word) for word in words])

    # # Scoring function to evaluate stemming
    # def evaluate_stemming(original_sentence, stemmed_sentence):
    #     """
    #     Score the quality of stemming based on semantic preservation and TF-IDF similarity.
    #     """
    #     # Lemmatized original sentence as the gold standard
    #     lemmatized_sentence = lemmatize_sentence(original_sentence)
        
    #     # TF-IDF similarity
    #     vectorizer = TfidfVectorizer()
    #     vectors = vectorizer.fit_transform([lemmatized_sentence, stemmed_sentence])
    #     similarity = (vectors * vectors.T).A[0, 1]
        
    #     # Penalty for reducing too much or too little
    #     reduction_ratio = len(stemmed_sentence) / len(original_sentence)
    #     if reduction_ratio < 0.5 or reduction_ratio > 0.9:
    #         similarity *= 0.8  # Penalize excessive or minimal stemming
        
    #     return similarity

    # # Dynamic function to detect the best stemming method
    # def detect_best_stemmer(sentence):
    #     stemmers = {
    #         "Porter": porter_stem(sentence),
    #         "Lancaster": lancaster_stem(sentence),
    #         "Snowball": snowball_stem(sentence)
    #     }
        
    #     scores = {
    #         name: evaluate_stemming(sentence, stemmed_sentence)
    #         for name, stemmed_sentence in stemmers.items()
    #     }
        
    #     best_stemmer = max(scores, key=scores.get)
    #     return best_stemmer, scores[best_stemmer], stemmers[best_stemmer]

    # # Example Usage
    # if __name__ == "__main__":
    #     sentence = "The children are playing happily in the gardens."
    #     best_stemmer, score, stemmed_sentence = detect_best_stemmer(sentence)
    #     print(f"Best Stemmer: {best_stemmer}\nScore: {score}\nStemmed Sentence: {stemmed_sentence}")



    # Streamlit UI
    # Placeholder for the combined_text (this would be filled elsewhere in your app)
    selected_sentences_source = {}
    selected_sentences_target = {}
    sentences = []

    # Extract raw sentences from source_content
    sentences.extend([
        source_content["file_content"]
        for source_content in selected_text_source.values()
    ])

    # Extract raw sentences from target_content
    sentences.extend([
        target_content["file_content"]
        for target_content in selected_text_target.values()
    ])

    # Ensure `sentences` is a list of individual strings

    # Create a mapping from sentence content to index before processing
    original_sentences = sentences[:]
    sentence_to_index = {sentence: idx for idx, sentence in enumerate(original_sentences)}

    # Streamlit Checkboxes for processing
    remove_stopwords_checkbox = st.checkbox("Remove Stopwords")
    processing_option = st.radio(
        "Choose Text Processing Method:",
        ("None", "Apply Lemmatization", "Apply Stemming")
    )

    # Process sentences while keeping their index intact
    if remove_stopwords_checkbox:
        stopwords = load_stopwords()
        sentences = [remove_stopwords(sentence, stopwords) for sentence in sentences]

    if processing_option == "Apply Lemmatization":
        sentences = [lemmatize_sentence(sentence) for sentence in sentences]

    if processing_option == "Apply Stemming":
        sentences = [stem_sentence(sentence) for sentence in sentences]



    # Display the processed sentences with their index
    




    tokens = preprocess_text(sentences)
    token_count = len(tokens)



    binary_matrix, occurrence_matrix = create_matrices(sentences ,tokens)
    binary_matrix_normlized_l1 = normalize_matrix_l1(binary_matrix)
    binary_matrix_normlized_l2 = normalize_matrix_l2(binary_matrix)
    occurrence_matrix_normalized_l1 = normalize_matrix_l1(occurrence_matrix)
    occurrence_matrix_normalized_l2 = normalize_matrix_l2(occurrence_matrix)



    binary_distance_manhattan = calculate_manhattan_distance(binary_matrix)
    binary_l1_distance_manhattan = calculate_manhattan_distance(binary_matrix_normlized_l1)
    binary_l2_distance_manhattan = calculate_manhattan_distance(binary_matrix_normlized_l2)
    occurrence_distance_manhattan = calculate_manhattan_distance(occurrence_matrix)
    occurrence_l1_distance_manhattan = calculate_manhattan_distance(occurrence_matrix_normalized_l1)
    occurrence_l2_distance_manhattan = calculate_manhattan_distance(occurrence_matrix_normalized_l2)


    # Calculate Euclidean distances for binary and occurrence matrices
    binary_distance_euclidean = calculate_euclidean_distance(binary_matrix)
    binary_l1_distance_euclidean = calculate_euclidean_distance(binary_matrix_normlized_l1)
    binary_l2_distance_euclidean = calculate_euclidean_distance(binary_matrix_normlized_l2)
    occurrence_distance_euclidean = calculate_euclidean_distance(occurrence_matrix)
    occurrence_l1_distance_euclidean = calculate_euclidean_distance(occurrence_matrix_normalized_l1)
    occurrence_l2_distance_euclidean = calculate_euclidean_distance(occurrence_matrix_normalized_l2)


    # Calculate Jaccard distances for binary and occurrence matrices
    binary_distance_jaccard = calculate_jaccard_distance(binary_matrix)
    binary_l1_distance_jaccard = calculate_jaccard_distance(binary_matrix_normlized_l1)
    binary_l2_distance_jaccard = calculate_jaccard_distance(binary_matrix_normlized_l2)
    occurrence_distance_jaccard = calculate_jaccard_distance(occurrence_matrix)
    occurrence_l1_distance_jaccard = calculate_jaccard_distance(occurrence_matrix_normalized_l1)
    occurrence_l2_distance_jaccard = calculate_jaccard_distance(occurrence_matrix_normalized_l2)



    # Calculate Hamming distances for binary and occurrence matrices
    binary_distance_hamming = calculate_hamming_distance(binary_matrix)
    binary_l1_distance_hamming = calculate_hamming_distance(binary_matrix_normlized_l1)
    binary_l2_distance_hamming = calculate_hamming_distance(binary_matrix_normlized_l2)
    occurrence_distance_hamming = calculate_hamming_distance(occurrence_matrix)
    occurrence_l1_distance_hamming = calculate_hamming_distance(occurrence_matrix_normalized_l1)
    occurrence_l2_distance_hamming = calculate_hamming_distance(occurrence_matrix_normalized_l2)


    # Calculate Bray-Curtis distances for binary and occurrence matrices
    binary_distance_bray_curtis = calculate_bray_curtis_distance(binary_matrix)
    binary_l1_distance_bray_curtis = calculate_bray_curtis_distance(binary_matrix_normlized_l1)
    binary_l2_distance_bray_curtis = calculate_bray_curtis_distance(binary_matrix_normlized_l2)
    occurrence_distance_bray_curtis = calculate_bray_curtis_distance(occurrence_matrix)
    occurrence_l1_distance_bray_curtis = calculate_bray_curtis_distance(occurrence_matrix_normalized_l1)
    occurrence_l2_distance_bray_curtis = calculate_bray_curtis_distance(occurrence_matrix_normalized_l2)


    # Calculate Cosine distances for binary and occurrence matrices
    binary_distance_cosine = calculate_cosine_distance(binary_matrix)
    binary_l1_distance_cosine = calculate_cosine_distance(binary_matrix_normlized_l1)
    binary_l2_distance_cosine = calculate_cosine_distance(binary_matrix_normlized_l2)
    occurrence_distance_cosine = calculate_cosine_distance(occurrence_matrix)
    occurrence_l1_distance_cosine = calculate_cosine_distance(occurrence_matrix_normalized_l1)
    occurrence_l2_distance_cosine = calculate_cosine_distance(occurrence_matrix_normalized_l2)


    # Calculate Kullback-Leibler distances for binary and occurrence matrices
    binary_distance_kl = calculate_kullback_leibler_distance(binary_matrix)
    binary_l1_distance_kl = calculate_kullback_leibler_distance(binary_matrix_normlized_l1)
    binary_l2_distance_kl = calculate_kullback_leibler_distance(binary_matrix_normlized_l2)
    occurrence_distance_kl = calculate_kullback_leibler_distance(occurrence_matrix)
    occurrence_l1_distance_kl = calculate_kullback_leibler_distance(occurrence_matrix_normalized_l1)
    occurrence_l2_distance_kl = calculate_kullback_leibler_distance(occurrence_matrix_normalized_l2)



    tfidf_binary_matrix, tfidf_occurrence_matrix, tfidf_prob_matrix = create_tfidf_matrices(sentences, tokens, binary_matrix, occurrence_matrix)

    # Calcul des distances Manhattan pour les matrices TF-IDF
    tfidf_binary_distance_manhattan = calculate_manhattan_distance(tfidf_binary_matrix)
    tfidf_occurrence_distance_manhattan = calculate_manhattan_distance(tfidf_occurrence_matrix)
    tfidf_prob_distance_manhattan = calculate_manhattan_distance(tfidf_prob_matrix)

    # Calcul des distances Euclidienne pour les matrices TF-IDF
    tfidf_binary_distance_euclidean = calculate_euclidean_distance(tfidf_binary_matrix)
    tfidf_occurrence_distance_euclidean = calculate_euclidean_distance(tfidf_occurrence_matrix)
    tfidf_prob_distance_euclidean = calculate_euclidean_distance(tfidf_prob_matrix)

    # Calcul des distances Cosinus pour les matrices TF-IDF
    tfidf_binary_distance_cosine = calculate_cosine_distance(tfidf_binary_matrix)
    tfidf_occurrence_distance_cosine = calculate_cosine_distance(tfidf_occurrence_matrix)
    tfidf_prob_distance_cosine = calculate_cosine_distance(tfidf_prob_matrix)

    # Calcul des distances Bray-Curtis pour les matrices TF-IDF
    tfidf_binary_distance_bray_curtis = calculate_bray_curtis_distance(tfidf_binary_matrix)
    tfidf_occurrence_distance_bray_curtis = calculate_bray_curtis_distance(tfidf_occurrence_matrix)
    tfidf_prob_distance_bray_curtis = calculate_bray_curtis_distance(tfidf_prob_matrix)

    # Calcul des distances Kullback-Leibler pour les matrices TF-IDF
    tfidf_binary_distance_kl = calculate_kullback_leibler_distance(tfidf_binary_matrix)
    tfidf_occurrence_distance_kl = calculate_kullback_leibler_distance(tfidf_occurrence_matrix)
    tfidf_prob_distance_kl = calculate_kullback_leibler_distance(tfidf_prob_matrix)

    # Calcul des distances Jaccard pour les matrices TF-IDF
    tfidf_binary_distance_jaccard = calculate_jaccard_distance(tfidf_binary_matrix)
    tfidf_occurrence_distance_jaccard = calculate_jaccard_distance(tfidf_occurrence_matrix)
    tfidf_prob_distance_jaccard = calculate_jaccard_distance(tfidf_prob_matrix)

    # Calcul des distances Hamming pour les matrices TF-IDF
    tfidf_binary_distance_hamming = calculate_hamming_distance(tfidf_binary_matrix)
    tfidf_occurrence_distance_hamming = calculate_hamming_distance(tfidf_occurrence_matrix)
    tfidf_prob_distance_hamming = calculate_hamming_distance(tfidf_prob_matrix)




    # Gestion des matrices en fonction des sélections
    if selected_descripteur == "Binaire" and selected_distance == "Euclidienne":
        label = "binary_distance_euclidean"
        choosen_matrix = binary_distance_euclidean
    elif selected_descripteur == "Binaire" and selected_distance == "Manhattan":
        label = "binary_distance_manhattan"
        choosen_matrix = binary_distance_manhattan
    elif selected_descripteur == "Binaire" and selected_distance == "Cosinus":
        label = "binary_distance_cosine"
        choosen_matrix = binary_distance_cosine
    elif selected_descripteur == "Binaire" and selected_distance == "Curtis":
        label = "binary_distance_bray_curtis"
        choosen_matrix = binary_distance_bray_curtis
    elif selected_descripteur == "Binaire" and selected_distance == "Leibler":
        label = "binary_distance_kl"
        choosen_matrix = binary_distance_kl
    elif selected_descripteur == "Binaire" and selected_distance == "Jaccard":
        label = "binary_distance_jaccard"
        choosen_matrix = binary_distance_jaccard
    elif selected_descripteur == "Binaire" and selected_distance == "Hamming":
        label = "binary_distance_hamming"
        choosen_matrix = binary_distance_hamming

    # Gestion des distances pour "Binaire normalisé L1"
    elif selected_descripteur == "Binaire normalise L1" and selected_distance == "Euclidienne":
        label = "binary_l1_distance_euclidean"
        choosen_matrix = binary_l1_distance_euclidean
    elif selected_descripteur == "Binaire normalise L1" and selected_distance == "Manhattan":
        label = "binary_l1_distance_manhattan"
        choosen_matrix = binary_l1_distance_manhattan
    elif selected_descripteur == "Binaire normalise L1" and selected_distance == "Cosinus":
        label = "binary_l1_distance_cosine"
        choosen_matrix = binary_l1_distance_cosine
    elif selected_descripteur == "Binaire normalise L1" and selected_distance == "Curtis":
        label = "binary_l1_distance_bray_curtis"
        choosen_matrix = binary_l1_distance_bray_curtis
    elif selected_descripteur == "Binaire normalise L1" and selected_distance == "Leibler":
        label = "binary_l1_distance_kl"
        choosen_matrix = binary_l1_distance_kl
    elif selected_descripteur == "Binaire normalise L1" and selected_distance == "Jaccard":
        label = "binary_l1_distance_jaccard"
        choosen_matrix = binary_l1_distance_jaccard
    elif selected_descripteur == "Binaire normalise L1" and selected_distance == "Hamming":
        label = "binary_l1_distance_hamming"
        choosen_matrix = binary_l1_distance_hamming

    # Gestion des distances pour "Binaire normalisé L2"
    elif selected_descripteur == "Binaire normalise L2" and selected_distance == "Euclidienne":
        label = "binary_l2_distance_euclidean"
        choosen_matrix = binary_l2_distance_euclidean
    elif selected_descripteur == "Binaire normalise L2" and selected_distance == "Manhattan":
        label = "binary_l2_distance_manhattan"
        choosen_matrix = binary_l2_distance_manhattan
    elif selected_descripteur == "Binaire normalise L2" and selected_distance == "Cosinus":
        label = "binary_l2_distance_cosine"
        choosen_matrix = binary_l2_distance_cosine
    elif selected_descripteur == "Binaire normalise L2" and selected_distance == "Curtis":
        label = "binary_l2_distance_bray_curtis"
        choosen_matrix = binary_l2_distance_bray_curtis
    elif selected_descripteur == "Binaire normalise L2" and selected_distance == "Leibler":
        label = "binary_l2_distance_kl"
        choosen_matrix = binary_l2_distance_kl
    elif selected_descripteur == "Binaire normalise L2" and selected_distance == "Jaccard":
        label = "binary_l2_distance_jaccard"
        choosen_matrix = binary_l2_distance_jaccard
    elif selected_descripteur == "Binaire normalise L2" and selected_distance == "Hamming":
        label = "binary_l2_distance_hamming"
        choosen_matrix = binary_l2_distance_hamming

    # Gestion des distances pour "Occurence"
    elif selected_descripteur == "Occurence" and selected_distance == "Euclidienne":
        label = "occurrence_distance_euclidean"
        choosen_matrix = occurrence_distance_euclidean
    elif selected_descripteur == "Occurence" and selected_distance == "Manhattan":
        label = "occurrence_distance_manhattan"
        choosen_matrix = occurrence_distance_manhattan
    elif selected_descripteur == "Occurence" and selected_distance == "Cosinus":
        label = "occurrence_distance_cosine"
        choosen_matrix = occurrence_distance_cosine
    elif selected_descripteur == "Occurence" and selected_distance == "Curtis":
        label = "occurrence_distance_bray_curtis"
        choosen_matrix = occurrence_distance_bray_curtis
    elif selected_descripteur == "Occurence" and selected_distance == "Leibler":
        label = "occurrence_distance_kl"
        choosen_matrix = occurrence_distance_kl
    elif selected_descripteur == "Occurence" and selected_distance == "Jaccard":
        label = "occurrence_distance_jaccard"
        choosen_matrix = occurrence_distance_jaccard
    elif selected_descripteur == "Occurence" and selected_distance == "Hamming":
        label = "occurrence_distance_hamming"
        choosen_matrix = occurrence_distance_hamming

    # Gestion des distances pour "Occurence normalisé L1"
    elif selected_descripteur == "Occurence Normalisé L1" and selected_distance == "Euclidienne":
        label = "occurrence_l1_distance_euclidean"
        choosen_matrix = occurrence_l1_distance_euclidean
    elif selected_descripteur == "Occurence Normalisé L1" and selected_distance == "Manhattan":
        label = "occurrence_l1_distance_manhattan"
        choosen_matrix = occurrence_l1_distance_manhattan
    elif selected_descripteur == "Occurence Normalisé L1" and selected_distance == "Cosinus":
        label = "occurrence_l1_distance_cosine"
        choosen_matrix = occurrence_l1_distance_cosine
    elif selected_descripteur == "Occurence Normalisé L1" and selected_distance == "Curtis":
        label = "occurrence_l1_distance_bray_curtis"
        choosen_matrix = occurrence_l1_distance_bray_curtis
    elif selected_descripteur == "Occurence Normalisé L1" and selected_distance == "Leibler":
        label = "occurrence_l1_distance_kl"
        choosen_matrix = occurrence_l1_distance_kl
    elif selected_descripteur == "Occurence Normalisé L1" and selected_distance == "Jaccard":
        label = "occurrence_l1_distance_jaccard"
        choosen_matrix = occurrence_l1_distance_jaccard
    elif selected_descripteur == "Occurence Normalisé L1" and selected_distance == "Hamming":
        label = "occurrence_l1_distance_hamming"
        choosen_matrix = occurrence_l1_distance_hamming

    # Gestion des distances pour "Occurence normalisé L2"
    elif selected_descripteur == "Occurence Normalisé L2" and selected_distance == "Euclidienne":
        label = "occurrence_l2_distance_euclidean"
        choosen_matrix = occurrence_l2_distance_euclidean
    elif selected_descripteur == "Occurence Normalisé L2" and selected_distance == "Manhattan":
        label = "occurrence_l2_distance_manhattan"
        choosen_matrix = occurrence_l2_distance_manhattan
    elif selected_descripteur == "Occurence Normalisé L2" and selected_distance == "Cosinus":
        label = "occurrence_l2_distance_cosine"
        choosen_matrix = occurrence_l2_distance_cosine
    elif selected_descripteur == "Occurence Normalisé L2" and selected_distance == "Curtis":
        label = "occurrence_l2_distance_bray_curtis"
        choosen_matrix = occurrence_l2_distance_bray_curtis
    elif selected_descripteur == "Occurence Normalisé L2" and selected_distance == "Leibler":
        label = "occurrence_l2_distance_kl"
        choosen_matrix = occurrence_l2_distance_kl
    elif selected_descripteur == "Occurence Normalisé L2" and selected_distance == "Jaccard":
        label = "occurrence_l2_distance_jaccard"
        choosen_matrix = occurrence_l2_distance_jaccard
    elif selected_descripteur == "Occurence Normalisé L2" and selected_distance == "Hamming":
        label = "occurrence_l2_distance_hamming"
        choosen_matrix = occurrence_l2_distance_hamming



    # TF-IDF Binaire
    if selected_descripteur == "TF-IDF Binaire" and selected_distance == "Euclidienne":
        label = "tfidf_binary_distance_euclidean"
        choosen_matrix = tfidf_binary_distance_euclidean

    if selected_descripteur == "TF-IDF Binaire" and selected_distance == "Manhattan":
        label = "tfidf_binary_distance_manhattan"
        choosen_matrix = tfidf_binary_distance_manhattan

    if selected_descripteur == "TF-IDF Binaire" and selected_distance == "Cosinus":
        label = "tfidf_binary_distance_cosine"
        choosen_matrix = tfidf_binary_distance_cosine

    if selected_descripteur == "TF-IDF Binaire" and selected_distance == "Curtis":
        label = "tfidf_binary_distance_bray_curtis"
        choosen_matrix = tfidf_binary_distance_bray_curtis

    if selected_descripteur == "TF-IDF Binaire" and selected_distance == "Leibler":
        label = "tfidf_binary_distance_kl"
        choosen_matrix = tfidf_binary_distance_kl

    if selected_descripteur == "TF-IDF Binaire" and selected_distance == "Jaccard":
        label = "tfidf_binary_distance_jaccard"
        choosen_matrix = tfidf_binary_distance_jaccard

    if selected_descripteur == "TF-IDF Binaire" and selected_distance == "Hamming":
        label = "tfidf_binary_distance_hamming"
        choosen_matrix = tfidf_binary_distance_hamming

    # TF-IDF Occurence
    if selected_descripteur == "TF-IDF Occurence" and selected_distance == "Euclidienne":
        label = "tfidf_occurrence_distance_euclidean"
        choosen_matrix = tfidf_occurrence_distance_euclidean

    if selected_descripteur == "TF-IDF Occurence" and selected_distance == "Manhattan":
        label = "tfidf_occurrence_distance_manhattan"
        choosen_matrix = tfidf_occurrence_distance_manhattan

    if selected_descripteur == "TF-IDF Occurence" and selected_distance == "Cosinus":
        label = "tfidf_occurrence_distance_cosine"
        choosen_matrix = tfidf_occurrence_distance_cosine

    if selected_descripteur == "TF-IDF Occurence" and selected_distance == "Curtis":
        label = "tfidf_occurrence_distance_bray_curtis"
        choosen_matrix = tfidf_occurrence_distance_bray_curtis

    if selected_descripteur == "TF-IDF Occurence" and selected_distance == "Leibler":
        label = "tfidf_occurrence_distance_kl"
        choosen_matrix = tfidf_occurrence_distance_kl

    if selected_descripteur == "TF-IDF Occurence" and selected_distance == "Jaccard":
        label = "tfidf_occurrence_distance_jaccard"
        choosen_matrix = tfidf_occurrence_distance_jaccard

    if selected_descripteur == "TF-IDF Occurence" and selected_distance == "Hamming":
        label = "tfidf_occurrence_distance_hamming"
        choosen_matrix = tfidf_occurrence_distance_hamming

    # TF-IDF Probabilité
    if selected_descripteur == "TF-IDF Probabilité" and selected_distance == "Euclidienne":
        label = "tfidf_prob_distance_euclidean"
        choosen_matrix = tfidf_prob_distance_euclidean

    if selected_descripteur == "TF-IDF Probabilité" and selected_distance == "Manhattan":
        label = "tfidf_prob_distance_manhattan"
        choosen_matrix = tfidf_prob_distance_manhattan

    if selected_descripteur == "TF-IDF Probabilité" and selected_distance == "Cosinus":
        label = "tfidf_prob_distance_cosine"
        choosen_matrix = tfidf_prob_distance_cosine

    if selected_descripteur == "TF-IDF Probabilité" and selected_distance == "Curtis":
        label = "tfidf_prob_distance_bray_curtis"
        choosen_matrix = tfidf_prob_distance_bray_curtis

    if selected_descripteur == "TF-IDF Probabilité" and selected_distance == "Leibler":
        label = "tfidf_prob_distance_kl"
        choosen_matrix = tfidf_prob_distance_kl

    if selected_descripteur == "TF-IDF Probabilité" and selected_distance == "Jaccard":
        label = "tfidf_prob_distance_jaccard"
        choosen_matrix = tfidf_prob_distance_jaccard

    if selected_descripteur == "TF-IDF Probabilité" and selected_distance == "Hamming":
        label = "tfidf_prob_distance_hamming"
        choosen_matrix = tfidf_prob_distance_hamming















    # Make sure to compare using original indices
    if not selected_sentences_source and not selected_text_source:
        st.error("Veuillez sélectionner des phrases ou des fichiers dans la colonne 'Quoi chercher'.")
    elif not selected_sentences_target and not selected_text_target:
        st.error("Veuillez sélectionner des phrases ou des fichiers dans la colonne 'Où chercher'.")
    else:
        st.write("### Résultats de la comparaison")

        # Compare entire files to sentences and vice versa
        for source_var, source_content in selected_text_source.items():
            for target_var, target_content in selected_text_target.items():
                # Get the original sentence content
                original_source_sentence = source_content["file_content"]
                original_target_sentence = target_content["file_content"]

                # Retrieve the original index from the mapping (before processing)
                source_index = sentence_to_index.get(original_source_sentence, None)
                target_index = sentence_to_index.get(original_target_sentence, None)

                if source_index is not None and target_index is not None:
                    # Get the distance from the choosen_matrix
                    distance = choosen_matrix[source_index][target_index]

                    # Display the comparison
                    st.markdown(f"""
                        Comparing **<span style='color:#FFD700'> {source_var}</span> 
                        <span style='color:#B0E57C'> : {source_content['sentences']}</span>**
                        <br> with <br>
                        **<span style='color:#50C878'>{target_var} : {target_content['sentences']}</span>**
                        <br> Distance: **<span style='color:#FF6347'>{distance}</span>**
                        """, unsafe_allow_html=True)

                else:
                    st.warning(f"Sentence mismatch: Could not find matching sentences for comparison.")





    import pandas as pd

    # Helper function to truncate and handle duplicates
    def label_sentences(sentences, max_len=400):
        truncated = [f"{sentence[:max_len]}..." if len(sentence) > 50 else sentence for sentence in sentences]
        counts = {}
        labeled = []
        
        for sentence in truncated:
            if sentence in counts:
                counts[sentence] += 1
                labeled.append(f"{sentence} ({counts[sentence]})")
            else:
                counts[sentence] = 1
                labeled.append(sentence)
        
        return labeled

    # Truncate and label sentences for uniqueness
    labeled_sentences = label_sentences(sentences, max_len=400)

    # Create the DataFrame with labeled sentences as row and column labels
    dataframe = pd.DataFrame(
        choosen_matrix, 
        columns=labeled_sentences,
        index=labeled_sentences
    )

    # Display the DataFrame as an interactive table
    st.dataframe(dataframe, use_container_width=True)

    from wordcloud import WordCloud
    import matplotlib.pyplot as plt

    # Combine all selected sentences from source and target into a single string
    combined_text = " ".join(
        [content["file_content"] for content in selected_text_source.values()] +
        [content["file_content"] for content in selected_text_target.values()]
    )

    # Check if there is text to generate a word cloud
    if combined_text.strip():
        # Generate the word cloud
        wordcloud = WordCloud(
            width=800, height=400, 
            background_color="white", 
            colormap="viridis"
        ).generate(combined_text)

        # Display the word cloud using Matplotlib
        st.markdown("## 🌟 Nuage de mots ")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")  # Remove axes
        st.pyplot(fig)
    else:
        st.warning("No text available to generate a word cloud. Please select some content.")

