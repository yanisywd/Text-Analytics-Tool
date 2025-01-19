
import streamlit as st

def corpus_to_corpus():
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


    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    # Compute similarity matrix based on TF-IDF




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


  
    def calculate_similarity_matrix(distance_matrix):
        max_distance = np.max(distance_matrix)
        return 1 - (distance_matrix / max_distance)
    # Calculer les matrices de similarité

    

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


    st.write(sentences)


    sentence_to_filename_target = {}

    # Iterate through selected_text_target to build the mapping
    for var_name, content in selected_text_target.items():
        # Get the list of sentences
        sentences = content.get("sentences", [])  # Default to an empty list if not found
        
        # Get the filename or use a default value if "original_file" is missing
        filename = content.get("original_file", f"Folder: {var_name}")  # Use the variable name for folder-level selection

        # Map each sentence to its filename (or folder name for directory-level)
        for sentence in sentences:
            sentence_to_filename_target[sentence] = filename

    



    tokens = preprocess_text(sentences)


    def compute_similarity_matrix(sentences):
        # Vectorize the sentences using TF-IDF
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(sentences)

        # Compute pairwise cosine similarity
        similarity_matrix = cosine_similarity(tfidf_matrix)

        return similarity_matrix

    # Ensure `sentences` is a list of individual strings
    if sentences:
        # Calculate the similarity matrix
        choosen_matrix = compute_similarity_matrix(sentences)
        st.success("Similarity matrix successfully calculated.")
    else:
        st.error("No sentences found to compute the similarity matrix.")

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


# Allow user to choose k
  # Allow user to input k
    st.success(f"Nombre de corpus : {len(sentences)}")

    k = st.number_input("Choose the number of nearest neighbors (k):", min_value=1, value=3, step=1)

    def compute_k_nearest_neighbors_for_source(similarity_matrix, source_sentences, target_sentences, k):
        """
        Computes the k nearest neighbors of each source sentence from the target sentences.
        Returns a dictionary where each source sentence maps to its k nearest neighbors.
        """
        neighbors = {}
        for source_idx, source_sentence in enumerate(source_sentences):
            # Get the similarity row for the source sentence
            similarities = similarity_matrix[source_idx]
            
            # Find the indices of the k highest similarities in target sentences
            top_k_indices = similarities.argsort()[-k:][::-1]
            top_k_values = similarities[top_k_indices]
            
            # Map the source sentence to its neighbors
            neighbors[source_sentence] = [
                {"target_sentence": target_sentences[target_idx], "similarity": similarity}
                for target_idx, similarity in zip(top_k_indices, top_k_values)
            ]
        return neighbors

    # Collect source and target sentences
    source_sentences = [content["file_content"] for content in selected_text_source.values()]
    target_sentences = [content["file_content"] for content in selected_text_target.values()]

    if source_sentences and target_sentences:
        # Combine source and target sentences for vectorization
        all_sentences = source_sentences + target_sentences

        # Compute the TF-IDF matrix
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(all_sentences)

        # Compute similarity between source and target
        source_matrix = tfidf_matrix[:len(source_sentences)]
        target_matrix = tfidf_matrix[len(source_sentences):]
        similarity_matrix = cosine_similarity(source_matrix, target_matrix)

        # Compute k-nearest neighbors
        k_neighbors = compute_k_nearest_neighbors_for_source(similarity_matrix, source_sentences, target_sentences, k)

        # Display results
# Display Results with Enhanced Formatting
    st.markdown("## 🟢 **k-Nearest Neighbors Results**")

    for source_sentence, neighbors in k_neighbors.items():
        st.markdown(f"""
        <div style="padding: 10px; border: 2px solid #4CAF50; border-radius: 10px; margin-bottom: 20px; background-color: #f9f9f9;">
            <p style="font-size: 16px; font-weight: bold; color: #4CAF50;">🔷 Source Sentence:</p>
            <p style="font-size: 15px; font-style: italic; color: #333;">"{source_sentence}"</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"<p style='color: #007BFF; font-size: 15px; font-weight: bold;'>Top {len(neighbors)} Neighbors:</p>", unsafe_allow_html=True)

        for i, neighbor in enumerate(neighbors, start=1):
            target_sentence = neighbor['target_sentence']

            file_or_folder = sentence_to_filename_target.get(target_sentence, "Unknown Source")

            st.markdown(f"""
            <div style="padding: 10px; margin-left: 20px; border-left: 3px solid #007BFF; background-color: #f2f6fc; margin-bottom: 10px;">
                <p style="font-size: 14px; color: #555; margin-bottom: 5px;">🔹 **Neighbor {i}:**</p>
                <p style="font-size: 14px; color: #333;">{neighbor['target_sentence']}</p>
                <p style="font-size: 14px; font-weight: bold; color: #FF5722;">Similarity: {neighbor['similarity']:.4f}</p>
                <p style="font-size: 14px; font-weight: bold; color: #4CAF50;">📄 Source: {file_or_folder}</p>
            </div>
            """, unsafe_allow_html=True)

    # Error handling for missing sentences





    # Convert k-NN results into a DataFrame for visualization
    nn_data = []
    for source_sentence, neighbors in k_neighbors.items():
        for neighbor in neighbors:
            nn_data.append({
                "Source Sentence": source_sentence,
                "Target Sentence": neighbor["target_sentence"],
                "Similarity": neighbor["similarity"]
            })

    # Create DataFrame
    df_neighbors = pd.DataFrame(nn_data)

    # Display DataFrame
    st.dataframe(df_neighbors, use_container_width=True)



    from wordcloud import WordCloud, STOPWORDS
    import matplotlib.pyplot as plt
    from sklearn.feature_extraction.text import TfidfVectorizer
    import streamlit as st
    from nltk.corpus import stopwords
    import nltk

    nltk.download('stopwords')

    # Streamlit setup
    st.title("Interactive Word Cloud Generator")
    st.sidebar.header("Configuration Options")

    # Predefined stopwords
    french_stopwords = set(stopwords.words('french'))
    english_stopwords = set(stopwords.words('english'))
    all_stopwords = STOPWORDS.union(french_stopwords).union(english_stopwords)

    # User inputs for customization
    background_color = st.sidebar.color_picker("Background Color", "#ffffff")
    colormap = st.sidebar.selectbox("Color Map", ["viridis", "plasma", "inferno", "magma", "Spectral", "coolwarm", "spring", "summer", "autumn", "winter", "cool", "Wistia"])
    max_words = st.sidebar.slider("Max Words", 50, 500, 300)
    display_mode = st.sidebar.radio("Display Mode", ["Default", "TF-IDF-based"])

    # Combine all selected sentences from source and target into a single string
    combined_text = " ".join(
        [content["file_content"] for content in selected_text_source.values()] +
        [content["file_content"] for content in selected_text_target.values()]
    )

    # Check if there is text to generate a word cloud
    if combined_text.strip():
        if display_mode == "TF-IDF-based":
            # Compute TF-IDF scores
            vectorizer = TfidfVectorizer(stop_words=list(all_stopwords), max_features=max_words)
            tfidf_matrix = vectorizer.fit_transform([combined_text])
            tfidf_scores = dict(zip(vectorizer.get_feature_names_out(), tfidf_matrix.toarray().flatten()))

            # Generate word cloud using TF-IDF scores
            wordcloud = WordCloud(
                width=800, 
                height=400, 
                background_color=background_color, 
                colormap=colormap
            ).generate_from_frequencies(tfidf_scores)
        else:
            # Generate default word cloud
            wordcloud = WordCloud(
                width=800, 
                height=400, 
                background_color=background_color, 
                colormap=colormap, 
                max_words=max_words, 
                stopwords=all_stopwords
            ).generate(combined_text)

        # Display the word cloud
        st.markdown("## 🌟 Generated Word Cloud")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")  # Remove axes
        st.pyplot(fig)
    else:
        st.warning("No text available to generate a word cloud. Please select some content.")





    import re
    from collections import Counter
    import streamlit as st
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    from PIL import Image

    # --- Helper Functions ---
    def extract_and_count_words(text):
        """Extracts and counts words from a given text."""
        words = re.findall(r'\b\w+\b', text.lower())
        stopwords = set(["le", "la", "les", "un", "une", "de", "du", "et", "en", "à", "the", "is", "in", "on", "for", "with"])
        words = [word for word in words if word not in stopwords]
        return Counter(words)

    def analyze_repeated_words(word_counter):
        """Filters words with frequency > 1."""
        return {word: count for word, count in word_counter.items() if count > 1}

    def visualize_wordcloud(word_counter):
        """Generates and returns a WordCloud figure."""
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(word_counter)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        return fig

    def visualize_histogram(repeated_words):
        """Generates and returns a histogram for the 10 most repeated words."""
        fig, ax = plt.subplots(figsize=(10, 5))
        if repeated_words:
            # Sort repeated words by frequency and take the top 10
            top_repeated_words = dict(sorted(repeated_words.items(), key=lambda x: x[1], reverse=True)[:10])
            
            ax.bar(top_repeated_words.keys(), top_repeated_words.values(), color='#4CAF50')
            ax.set_title("Top 10 Most Repeated Words", fontsize=16)
            ax.set_xlabel("Words", fontsize=12)
            ax.set_ylabel("Frequency", fontsize=12)
            ax.tick_params(axis='x', rotation=45)
        else:
            ax.text(0.5, 0.5, "No repeated words found", fontsize=14, ha="center", va="center", alpha=0.6)
            ax.axis("off")
        return fig

    # --- Streamlit App ---

    # Header
    st.markdown(
        """
        <style>
            .title {
                text-align: center;
                font-size: 2.5em;
                color: #4CAF50;
                margin-bottom: 20px;
            }
            .subheader {
                text-align: center;
                font-size: 1.5em;
                color: #333;
                margin-bottom: 10px;
            }
            .card {
                background-color: #f9f9f9;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
            }
        </style>
        """, unsafe_allow_html=True)



    # --- Text Input (Direct Display) ---



    # --- Analysis ---
    word_counter = extract_and_count_words(combined_text)
    repeated_words = analyze_repeated_words(word_counter)

    # --- Visualizations ---
# --- Visualizations ---
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.write("### Histogram of Repeated Words")
    fig_hist = visualize_histogram(repeated_words)
    st.pyplot(fig_hist)
    st.markdown('</div>', unsafe_allow_html=True)



    # --- Frequent Words ---
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.write("### Top 10 Most Frequent Words:")
    top_words = word_counter.most_common(10)
    for word, count in top_words:
        st.write(f"**{word.capitalize()}**: {count}")
    st.markdown('</div>', unsafe_allow_html=True)

# --- Repeated Words ---
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.write("### Repeated Words:")

    if repeated_words:
        # Sort repeated words by count
        repeated_words_sorted = sorted(repeated_words.items(), key=lambda x: x[1], reverse=True)
        
        # Display the top 5 repeated words
        st.write("**Top 5 Repeated Words:**")
        for word, count in repeated_words_sorted[:5]:
            st.write(f"**{word.capitalize()}**: {count}")
        
        # Expander to show more words
        if len(repeated_words_sorted) > 5:
            with st.expander("Show More Repeated Words"):
                st.write("**All Repeated Words:**")
                for word, count in repeated_words_sorted[5:]:
                    st.write(f"**{word.capitalize()}**: {count}")
    else:
        st.write("No repeated words found.")
    st.markdown('</div>', unsafe_allow_html=True)

