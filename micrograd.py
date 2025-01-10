def home_page():
    import re
    import streamlit as st



    label = None



    import streamlit as st

    # Custom CSS
    custom_css = """
    <style>
    /* Sidebar background and text color */
    div[data-testid="stSidebar"] {
        background-color: #001f4d; /* Deep blue */
        color: white; /* White text */
    }
    div[data-testid="stSidebar"] h2, div[data-testid="stSidebar"] p, div[data-testid="stSidebar"] a {
        color: white; /* Ensure all text, including links, is white */
    }

    /* Main screen background and text color */
    div[data-testid="stAppViewContainer"] {
        background-color: #001f4d; /* Deep blue */
        
    }

    /* White scrollbar for better contrast */
    ::-webkit-scrollbar {
        width: 10px;
    }

    ::-webkit-scrollbar-thumb {
        background-color: #f0f0f0; /* White scrollbar */
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background-color: #cfcfcf; /* Slightly darker hover */
    }
    </style>
    """

    # Inject custom CSS
    st.markdown(custom_css, unsafe_allow_html=True)




    # Function to split text into sentences and tokens
    def split_text_into_sentences_and_tokens(text):
        # Clean and split the text into sentences using punctuation (.,!?)
        text = text.replace('�', '')
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        
        # Remove punctuation from sentences
        sentences_no_punctuation = [re.sub(r'[.,!-?;\'"]', '', sentence) for sentence in sentences]
        
        # Create a set to store tokens and avoid duplicates
        token_set = set()
        
        for sentence in sentences_no_punctuation:
            # Extract words only, excluding punctuation
            tokens = re.findall(r'\b\w+\b', sentence)
            token_set.update(token.lower() for token in tokens)
        
        # Convert the set to a list of unique tokens
        tokens = list(token_set)
        
        return sentences_no_punctuation, tokens



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





    import streamlit as st

    def read_file_content(uploaded_file):
        """Reads the content of an uploaded text file."""
        encodings = ['utf-8', 'utf-16', 'latin-1']
        for encoding in encodings:
            try:
                return uploaded_file.read().decode(encoding)
            except UnicodeDecodeError:
                continue
        return None

    # Streamlit UI
    st.sidebar.title("Note :Uploader des documents pour afficher les options:")

    # Field 1: Single Text File Upload
    modified_text_one_file = ""

    st.subheader("Choisir le fichier qui contient vos documents")
    single_uploaded_file = st.file_uploader("Choose a single text file", type="txt")
    single_file_text = ""
    if single_uploaded_file:
        single_file_text = read_file_content(single_uploaded_file)
        st.write("Single File Content:")
        # Affiche le texte dans une zone modifiable
        modified_text_one_file = st.text_area("Edit the Text", single_file_text, height=200)




    # modified_text_one_file









    import streamlit as st
    import os

    modified_text_multiplefiles = ""

    # Define the paths to the main folders
    MAIN_FOLDERS = ["/Users/yanis/Desktop/yahiaoui/tf-idf_app/Corpus_Anglais", "/Users/yanis/Desktop/yahiaoui/tf-idf_app/Corpus_Francais"]

    # Function to get all .txt files in the given main folders
    def get_all_txt_files(folders):
        txt_files = []
        for folder in folders:
            for root, dirs, files in os.walk(folder):
                for file in files:
                    if file.endswith(".txt"):
                        full_path = os.path.join(root, file)
                        txt_files.append(full_path)
        return txt_files

    # Function to read and concatenate the content of selected files
    def concatenate_selected_files(selected_files):
        concatenated_text = ""
        for file_path in selected_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    concatenated_text += f.read() + "\n"
            except Exception as e:
                st.error(f"Error reading {file_path}: {e}")
        return concatenated_text

    # Main Streamlit app
    st.title("Choisir où vous voulez chercher ")

    # Get all .txt files from the main folders
    all_txt_files = get_all_txt_files(MAIN_FOLDERS)

    # Let the user select multiple files from the list
    selected_files = st.multiselect("Choose files from predefined folders:", all_txt_files)

    # Allow the user to upload their own file
    uploaded_file = st.file_uploader("Or upload your own file", type=["txt"])

    # Process uploaded file
    if uploaded_file is not None:
        try:
            # Read the uploaded file's content
            uploaded_text = uploaded_file.read().decode("utf-8")
            
            # Add the uploaded file content to the selected files
            concatenated_text = ""
            if selected_files:
                concatenated_text = concatenate_selected_files(selected_files)
            concatenated_text += uploaded_text + "\n"

            # Display the concatenated text in a text area and allow the user to edit it
            modified_text_multiplefiles = st.text_area("Edit the Concatenated Text", concatenated_text, height=300)
        except Exception as e:
            st.error(f"Error processing uploaded file: {e}")
    elif selected_files:
        # Concatenate and display the text from selected files
        concatenated_text = concatenate_selected_files(selected_files)
        modified_text_multiplefiles = st.text_area("Edit the Concatenated Text", concatenated_text, height=300)
    else:
        st.info("No files selected or uploaded.")



    #modified_text_multiplefiles





    # # Assuming `concatenated_text` is the variable containing all concatenated text
    # sentences = split_into_sentences(modified_text_multiplefiles)

    # # Create a list of indexed sentences for selection
    # indexed_sentences = [f"{i + 1}: {sentence}" for i, sentence in enumerate(sentences)]

    # # Streamlit field for selecting a sentence
    # selected_sentence = st.selectbox("Select a sentence:", indexed_sentences)

    # # Display the chosen sentence
    # st.write(f"**Selected Sentence:** {selected_sentence.split(': ', 1)[1]}")




    import spacy

    # Load the small English model
    nlp = spacy.load("en_core_web_sm")



    combined_text = modified_text_one_file + "\n" + modified_text_multiplefiles




    import streamlit as st
    import spacy
    from nltk.stem import PorterStemmer

    # Define the path to the stopwords file
    STOPWORDS_FILE = "/Users/yanis/Desktop/yahiaoui/tf-idf_app/stopwords.txt"

    # Function to load stopwords from a file
    def load_stopwords():
        try:
            with open(STOPWORDS_FILE, "r", encoding="utf-8") as f:
                stopwords = f.read().splitlines()
            return set(stopwords)
        except Exception as e:
            st.error(f"Error loading stopwords: {e}")
            return set()

    # Function to remove stopwords from the given text
    def remove_stopwords(text, stopwords):
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in stopwords]
        return " ".join(filtered_words)

    # Initialize spaCy and NLTK
    nlp = spacy.load("en_core_web_sm")
    stemmer = PorterStemmer()

    # Lemmatization function (using spaCy)
    def lemmatize_text(text):
        doc = nlp(text)
        lemmatized_text = " ".join([token.lemma_ for token in doc])
        return lemmatized_text

    # Stemming function (using NLTK)
    def stem_text(text):
        words = text.split()
        stemmed_text = " ".join([stemmer.stem(word) for word in words])
        return stemmed_text

    # Streamlit UI

    # Placeholder for the combined_text (this would be filled elsewhere in your app)


    # Display the original combined text
    st.subheader("Text avant modifications:")
    st.text_area("Combined Text", combined_text, height=200)

    # Checkbox for removing stopwords
    remove_stopwords_checkbox = st.checkbox("Remove Stopwords")

    # Checkbox for lemmatization
    apply_lemmatization = st.checkbox("Apply Lemmatization")

    # Checkbox for stemming
    apply_stemming = st.checkbox("Apply Stemming")

    # Process text if checkboxes are checked
    if remove_stopwords_checkbox:
        stopwords = load_stopwords()
        combined_text = remove_stopwords(combined_text, stopwords)

    if apply_lemmatization:
        combined_text = lemmatize_text(combined_text)

    if apply_stemming:
        combined_text = stem_text(combined_text)

    # Display the processed text
    st.subheader("Text apres modifications:")
    st.text_area("Processed Text", combined_text, height=200)








    if combined_text:
        sentences = split_into_sentences(combined_text)
        st.write(f"les sentences : {sentences}")
        tokens = preprocess_text(sentences)
        token_count = len(tokens)
        sentences_no_punctuation_count = len(sentences)

    print(sentences_no_punctuation_count , token_count)






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








    selected_descripteur = st.sidebar.selectbox('Descripteur', ('Binaire','Occurence','Binaire normalise L1','Binaire normalise L2','Occurence Normalisé L1', 'Occurence Normalisé L2', 'TF-IDF Binaire', 
    'TF-IDF Occurence', 'TF-IDF Probabilité'))

    selected_distance = st.sidebar.selectbox('Distance', ('Euclidienne', 'Manhattan', 'Cosinus','Curtis','Leibler','Jaccard','Hamming'))







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



    # Ensure the similarity matrix is calculated based on the choosen_matrix
    if combined_text:  # Make sure there is text to work with
        # Calculate the similarity matrix using the chosen matrix (replace 'choosen_matrix' with your actual matrix)
        similarity_matrix = calculate_similarity_matrix(choosen_matrix)

        # Sidebar for K nearest documents
        st.sidebar.header("K Nearest Documents : ")

        # Slider for selecting K
        k = st.sidebar.slider("Select K", min_value=1, max_value=len(sentences), value=2)

        # Dropdown for selecting the document
        document_options = ["Whole Corpus"] + [f"Document {i + 1}: {sentences[i][:300]}" for i in range(len(sentences))]
        selected_document = st.sidebar.selectbox("Select Document", document_options)

        # Sidebar for custom query input
        st.sidebar.subheader("Or Enter Your Own Phrase:")
        custom_query = st.sidebar.text_area("Enter a custom query to search against the corpus (optional):", height=100)

        # Determine the query source (custom query or selected document)
        if custom_query.strip():  # If the user entered a custom query
            selected_query = custom_query.strip()
            doc_requete_index = None  # No pre-selected document
        else:  # Use the selected document from the dropdown
            selected_query = None
            if selected_document == "Whole Corpus":
                doc_requete_index = 0  # Assuming the whole corpus is indexed as 0
            else:
                import re
                match = re.search(r"Document (\d+):", selected_document)
                doc_requete_index = int(match.group(1)) - 1 if match else 0  # Convert to zero-based index

        # Button to execute the function
        if st.sidebar.button("Find K Nearest Documents"):
            if selected_query:  # Custom query entered
                nearest_docs = k_plus_proches_documents_custom_query(selected_query, sentences, k)
            elif doc_requete_index is not None:  # Pre-selected document
                nearest_docs = k_plus_proches_documents(doc_requete_index, similarity_matrix, k)
            else:
                nearest_docs = []

            # Display results in an expander for better organization
            if nearest_docs:  # Check if there are results
                with st.expander(f"Les K plus proches Documents:", expanded=True):
                    if selected_query:
                        st.markdown(
                            f"<strong style='color:green;'>Phrase/Query choisie :</strong> "
                            f"<strong style='color:orange;'>{custom_query}</strong>",
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f"<strong style='color:green;'>Document choisi :</strong> "
                            f"<strong style='color:orange;'>{selected_document}</strong>",
                            unsafe_allow_html=True
                        )
                    for idx, score in nearest_docs:
                        st.markdown(
                            f"<strong>Document {idx + 1}:</strong> {sentences[idx]} "
                            f"<span style='color: #FF5733; font-weight: bold;'> (Similarité: {score:.4f})</span>",
                            unsafe_allow_html=True
                        )
            else:
                st.write("Aucun document similaire trouvé.")






    import streamlit as st
    import pandas as pd
    import numpy as np



    with st.expander("Sélectionnez deux documents pour voir leur similarité (La distance qui les séparent):", expanded=True):
        # Combine document number and content for display
        doc_options = [f'Doc {i + 1}: {sentence}' for i, sentence in enumerate(sentences)]

        # Let the user select two documents
        selected_docs = st.multiselect("Choisissez deux documents", options=doc_options, max_selections=2)

        # Check if exactly two documents are selected
        if len(selected_docs) == 2:
            # Get indices of selected documents
            doc1_index = int(selected_docs[0].split(' ')[1].strip(':')) - 1
            doc2_index = int(selected_docs[1].split(' ')[1].strip(':')) - 1

            # Retrieve the similarity score from the matrix
            similarity_score = choosen_matrix[doc1_index, doc2_index]

            # Display the results with improved formatting and colored titles
            st.markdown(
                f"""
                <div style="font-size: 16px; color: #1F77B4; font-weight: bold;">
                    Document 1:
                </div>
                <div style="margin-bottom: 10px;">
                    {selected_docs[0]}
                </div>

                <div style="font-size: 16px; color: #1F77B4; font-weight: bold;">
                    Document 2:
                </div>
                <div style="margin-bottom: 10px;">
                    {selected_docs[1]}
                </div>

                <div style="font-size: 18px; color: #D62728; font-weight: bold; margin-top: 20px;">
                Score Similarité: {similarity_score:.4f}
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.write("Veuillez sélectionner exactement deux documents.")




    def format_label(label):
        # Remove underscores, capitalize each word, and add 'DF' at the end
        formatted_label = ' '.join(word.capitalize() for word in label.split('_'))
        return f"{formatted_label} DF : "


    # Format the label for display
    formatted_label = format_label(label)


    st.markdown(
        f'<p style="color:orange;">{formatted_label}</p>',
        unsafe_allow_html=True
    )

    # Custom CSS for a compact table
    st.markdown(
        """
        <style>
        .stTable table {
            font-size: 12px;  /* Adjust font size */
            padding: 4px;     /* Reduce padding */
            width: 60%;       /* Adjust width as needed */
        }
        .stTable th, .stTable td {
            padding: 6px 8px; /* Make table cells more compact */
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Display the similarity matrix as a compact table
    dataframe = pd.DataFrame(
        choosen_matrix, 
        columns=[f'Doc {i + 1}' for i in range(len(sentences))],
        index=[f'Doc {i + 1}' for i in range(len(sentences))]
    )
    st.table(dataframe)




    import streamlit as st
    import matplotlib.pyplot as plt
    from collections import Counter

    # Function to plot word frequency
    def plot_word_frequencies(text):
        words = text.split()
        word_counts = Counter(words)
        common_words = word_counts.most_common(10)  # Top 10 words
        labels, values = zip(*common_words)
        
        plt.figure(figsize=(10, 5))
        plt.bar(labels, values)
        plt.xlabel('Words')
        plt.ylabel('Frequency')
        plt.title('Top 10 Most Frequent Words')
        st.sidebar.pyplot(plt)  # Display in the sidebar

    # Sample combined text for testing

    plot_word_frequencies(combined_text)

 












#-------------------------------------------------------
def contact_page():
    st.title("Contact Page")
    st.write("Feel free to reach out through the Contact page.")

# Mapping of page names to page functions
pages = {
    "Home": about_page,
    "About": about_page,
    "Contact": contact_page,
}

# Initialize session state for the current page
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "Home"

# Define the navbar as a radio button (or other Streamlit widget)
with st.sidebar:
    st.title("Navigation")
    st.session_state["current_page"] = st.radio("Go to", list(pages.keys()))

# Render the content of the selected page
page_function = pages[st.session_state["current_page"]]
page_function()









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
        return sorted(unique_tokens)




# Define main folders
    MAIN_FOLDERS = [
        "/Users/yanis/Desktop/yahiaoui/tf-idf_app/Corpus_Anglais",
        "/Users/yanis/Desktop/yahiaoui/tf-idf_app/Corpus_Francais"
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
        selected_text_source = handle_selection(combined_corpus, MAIN_FOLDERS, "Quoi chercher")

    # Right column: "Où chercher ?"
    with col2:
        st.header("Où chercher ?")
        selected_text_target = handle_selection(combined_corpus, MAIN_FOLDERS, "Où chercher ?")

    # Display variables for debugging
    st.write("## Selected Variables (Source)")
    for var_name, content in selected_text_source.items():
        st.write(f"**{var_name}:** {content['file_content'][:100]}...")  # Display the first 100 characters for preview

    st.write("## Selected Variables (Target)")
    for var_name, content in selected_text_target.items():
        st.write(f"**{var_name}:** {content['file_content'][:100]}...")  # Display the first 100 characters for preview

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
    tokens = preprocess_text(sentences)


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
                    Comparing **<span style='color:blue'>{source_var} : {source_content['sentences']}</span>**
                    \n with \n **<span style='color:green'>{target_var} : {target_content['sentences']}</span>**
                    \n Distance: **<span style='color:red'>{distance}</span>**
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
