
import streamlit as st

def document_to_document():
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
    # st.write("## Selected Variables (Source)")
    # for var_name, content in selected_text_source.items():
    #     st.write(f"**{var_name}:** {content['file_content'][:100]}...")  # Display the first 100 characters for preview

    # st.write("## Selected Variables (Target)")
    # for var_name, content in selected_text_target.items():
    #     st.write(f"**{var_name}:** {content['file_content'][:100]}...")  # Display the first 100 characters for preview







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


# Build the sentence-to-filename mapping
# Create a mapping for target sentences to filenames
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

    
    
    # st.write(sentence_to_filename_target)




    tokens = preprocess_text(sentences)


    def compute_similarity_matrix(sentences):
        # Vectorize the sentences using TF-IDF
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(sentences)

        # Compute pairwise cosine similarity
        similarity_matrix = cosine_similarity(tfidf_matrix)

        return similarity_matrix

    # Ensure `sentences` is a list of individual strings
# Build the sentence_to_index mapping from the sentences list
    sentence_to_index = {sentence: idx for idx, sentence in enumerate(sentences)}

    # Compute similarity matrix if sentences exist
    if sentences:
        choosen_matrix = compute_similarity_matrix(sentences)
        st.success("Similarity matrix successfully calculated.")
    else:
        st.error("No sentences found to compute the similarity matrix.")

    # Compare sentences using the similarity matrix
    if not selected_sentences_source and not selected_text_source:
        st.error("Veuillez sélectionner des phrases ou des fichiers dans la colonne 'Quoi chercher'.")
    elif not selected_sentences_target and not selected_text_target:
        st.error("Veuillez sélectionner des phrases ou des fichiers dans la colonne 'Où chercher'.")
    else:
        st.write("### Résultats de la comparaison")

        for source_var, source_content in selected_text_source.items():
            for target_var, target_content in selected_text_target.items():
                original_source_sentence = source_content["file_content"]
                original_target_sentence = target_content["file_content"]

                # Retrieve indices
                source_index = sentence_to_index.get(original_source_sentence)
                target_index = sentence_to_index.get(original_target_sentence)

                # Validate indices and calculate similarity
                if source_index is not None and target_index is not None:
                    distance = choosen_matrix[source_index][target_index]
                    st.write(f"Similarity between source and target: {distance:.4f}")




# Allow user to choose k
  # Allow user to input k
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

# Collect source sentences
    source_sentences = [
        sentence 
        for content in selected_text_source.values() 
        for sentence in split_into_sentences(content["file_content"])
    ]

    # Collect target sentences
    target_sentences = [
        sentence 
        for content in selected_text_target.values() 
        for sentence in split_into_sentences(content["file_content"])
    ]


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
            # Retrieve the sentence
            target_sentence = neighbor['target_sentence']

            # Find the corresponding file/folder using the mapping
            file_or_folder = sentence_to_filename_target.get(target_sentence, "Unknown Source")

            # Display the neighbor with the source file or folder
            st.markdown(f"""
            <div style="padding: 10px; margin-left: 20px; border-left: 3px solid #007BFF; background-color: #f2f6fc; margin-bottom: 10px;">
                <p style="font-size: 14px; color: #555; margin-bottom: 5px;">🔹 **Neighbor {i}:**</p>
                <p style="font-size: 14px; color: #333;">{target_sentence}</p>
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


    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    from collections import Counter
    import pandas as pd
    import seaborn as sns

    # Combine all selected sentences from source and target into a single string
    combined_text = " ".join(
        [content["file_content"] for content in selected_text_source.values()] +
        [content["file_content"] for content in selected_text_target.values()]
    )

    # Check if there is text to process
    if combined_text.strip():
        # Generate the word cloud
        wordcloud = WordCloud(
            width=800, height=400,
            background_color="white",
            colormap="viridis"
        ).generate(combined_text)

        # Display the word cloud using Matplotlib
        st.markdown("## 🌟 **Nuage de mots**")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")  # Remove axes
        st.pyplot(fig)

        # Tokenize and count word frequencies
        words = combined_text.lower().split()  # Split by spaces and convert to lowercase
        word_counts = Counter(words)

        # Get the 10 most frequent words
        most_common_words = word_counts.most_common(10)

        # Convert to a DataFrame for easy visualization
        word_freq_df = pd.DataFrame(most_common_words, columns=["Mot", "Fréquence"])

        # Plot a bar chart with Seaborn
        st.markdown("## 📊 **Top 10 des mots les plus fréquents**")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(
            x="Fréquence", 
            y="Mot", 
            data=word_freq_df,
            palette="viridis"
        )
        ax.set_title("Top 10 des mots les plus fréquents", fontsize=16, color="#4CAF50", pad=15)
        ax.set_xlabel("Fréquence", fontsize=14)
        ax.set_ylabel("Mot", fontsize=14)
        sns.despine(left=True, bottom=True)

        # Display the chart in Streamlit
        st.pyplot(fig)
    else:
        st.warning("Aucun texte disponible pour générer un nuage de mots ou un histogramme. Veuillez sélectionner du contenu.")


