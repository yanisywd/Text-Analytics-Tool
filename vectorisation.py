def vectorisation():
    import streamlit as st
    import gensim.downloader as api
    import fasttext
    import fasttext.util
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    import os
    import re
    from gensim.models import KeyedVectors, Doc2Vec
    from gensim.models.doc2vec import TaggedDocument

    # Load pre-trained Word2Vec, FastText, GloVe, and Doc2Vec models
    @st.cache_resource  # Cache the model to avoid reloading it multiple times
    def load_word2vec_model():
        return api.load('word2vec-google-news-300')

    @st.cache_resource  # Cache the model to avoid reloading it multiple times
    def load_fasttext_model():
        fasttext.util.download_model('en', if_exists='ignore')  # English model
        return fasttext.load_model('cc.en.300.bin')

    @st.cache_resource  # Cache the model to avoid reloading it multiple times
    def load_glove_model():
        glove_file = '/Users/yanis/Desktop/yahiaoui-app/Text-Analytics-Tool/glove.6B.200d.txt'  # Path to the GloVe file
        return KeyedVectors.load_word2vec_format(glove_file, no_header=True)

    @st.cache_resource  # Cache the model to avoid reloading it multiple times
    def load_doc2vec_model():
        # Assuming you have a pre-trained Doc2Vec model saved as 'doc2vec.model'
        return Doc2Vec.load('word2vec-google-news-300')

    # Select model type
    model_type = st.selectbox("Choose model type:", ["word2vec", "fasttext", "glove", "doc2vec"])

    # Load the selected model
    if model_type == "word2vec":
        model = load_word2vec_model()
    elif model_type == "fasttext":
        model = load_fasttext_model()
    elif model_type == "glove":
        model = load_glove_model()
    elif model_type == "doc2vec":
        model = load_doc2vec_model()

    # Preprocess text into tokens
    def preprocess_text(text):
        return re.findall(r'\b\w+\b', text.lower())

    # Convert sentences into vectors using Word2Vec, FastText, GloVe, or Doc2Vec
    def sentence_to_vector(sentence, model):
        tokens = preprocess_text(sentence)
        vectors = []

        if model_type in ["word2vec", "glove"]:
            # Word2Vec or GloVe model lookup
            vectors = [model[word] for word in tokens if word in model]
        elif model_type == "fasttext":
            # FastText model lookup
            vectors = [model.get_word_vector(word) for word in tokens]
        elif model_type == "doc2vec":
            # Doc2Vec model lookup
            doc = TaggedDocument(words=tokens, tags=[sentence])
            vectors = [model.infer_vector(doc.words)]

        if vectors:
            return np.mean(vectors, axis=0)  # Average the word vectors
        else:
            return np.zeros(model.vector_size)  # Return a zero vector if no tokens are in the model

    # Compute similarity matrix
    def compute_similarity_matrix(sentences, model):
        sentence_vectors = [sentence_to_vector(sentence, model) for sentence in sentences]
        return cosine_similarity(sentence_vectors)

    # Function to list the folder structure (arborescence)
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

    # Function to read file content
    def read_file_content(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                return file.read()
        except Exception as e:
            return f"Error reading {file_path}: {e}"

    # Function to split the text into sentences
    def split_into_sentences(text):
        return re.split(r'(?<=[.!?])\s+', text.strip())

    # Collect text from selected files and directories
    def collect_text_from_directory(base_path, relative_path, selected_items):
        full_path = os.path.join(base_path, relative_path)
        for root, dirs, files in os.walk(full_path):
            files = [f for f in files if f != ".DS_Store"]
            for file in files:
                file_path = os.path.join(root, file)
                content = read_file_content(file_path)
                sentences = split_into_sentences(content)
                variable_name = f"file_{os.path.relpath(file_path, base_path).replace('/', '_').replace(' ', '_')}"
                selected_items[variable_name] = {
                    "file_content": content,
                    "sentences": sentences
                }

    # Handle file and sentence selection
    def handle_selection(tree, base_paths, side):
        selected_items = {}
        st.write(f"### {side}")

        for folder, contents in tree.items():
            folder_selected = st.checkbox(f"📂 {folder}", key=f"{side}_folder_{folder}")

            if folder_selected:
                for base_path in base_paths:
                    full_path = os.path.join(base_path, folder)
                    if os.path.exists(full_path):
                        collect_text_from_directory(base_path, folder, selected_items)
                        break

            with st.expander(f"📂 {folder}"):
                for file in contents["files"]:
                    file_path = None
                    for base_path in base_paths:
                        potential_path = os.path.join(base_path, folder, file)
                        if os.path.exists(potential_path):
                            file_path = potential_path
                            break

                    if not file_path:
                        st.warning(f"Could not find file: {file}")
                        continue

                    file_selected = st.checkbox(f"📄 {file}", key=f"{side}_file_{file_path}")

                    if file_selected:
                        file_content = read_file_content(file_path)
                        sentences = split_into_sentences(file_content)

                        st.write("Select specific sentences:")
                        file_sentences_selected = False

                        for i, sentence in enumerate(sentences):
                            sentence_selected = st.checkbox(
                                f"Sentence {i+1}: {sentence}",
                                key=f"{side}_sentence_{file_path}_{i}"
                            )

                            if sentence_selected:
                                file_sentences_selected = True
                                variable_name = f"file_{file.replace('/', '_').replace(' ', '_')}_sentence_{i}"
                                selected_items[variable_name] = {
                                    "file_content": sentence,
                                    "sentences": [sentence],
                                    "original_file": file_path,
                                    "sentence_number": i
                                }

                        if not file_sentences_selected:
                            variable_name = f"file_{file.replace('/', '_').replace(' ', '_')}"
                            selected_items[variable_name] = {
                                "file_content": file_content,
                                "sentences": sentences,  # Ensure sentences are included
                                "original_file": file_path
                            }

        return selected_items

    # Main Streamlit UI
    st.title("Calcule de similarité avec vectorisation")

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
        manual_source_sentence = st.text_area("Entrer la phrase a chercher manuellement :", "")
        if manual_source_sentence:
            selected_text_source = {"manual_input": {"file_content": manual_source_sentence, "sentences": [manual_source_sentence]}}
        else:
            selected_text_source = handle_selection(combined_corpus, MAIN_FOLDERS, "Quoi chercher")

    # Right column: "Où chercher ?"
    with col2:
        st.header("Où chercher ?")
        selected_text_target = handle_selection(combined_corpus, MAIN_FOLDERS, "Où chercher ?")

    # Collect sentences from the selected sources and targets
    sentences = []

    # Process source text
    for content in selected_text_source.values():
        sentences.extend(content["sentences"])  # Assume sentences are already provided

    # Process target text
    for content in selected_text_target.values():
        if isinstance(content["file_content"], str):  # Check if it's a single paragraph
            # Split the paragraph into sentences and add to the list
            sentences.extend(split_into_sentences(content["file_content"]))
        elif isinstance(content["sentences"], list):  # If already split into sentences
            sentences.extend(content["sentences"])

    # Compute similarity matrix
    if sentences:
        similarity_matrix = compute_similarity_matrix(sentences, model)
        st.success("Similarity matrix successfully calculated.")
    else:
        st.error("No sentences found to compute the similarity matrix.")

    # Display results
    st.success(f"Nombre de Phrases : {len(sentences)}")
    st.write("### Résultats de la similarité au niveau des phrases")
    for source_var, source_content in selected_text_source.items():
        source_sentences = source_content["sentences"]  # Get split sentences from source
        for target_var, target_content in selected_text_target.items():
            target_sentences = target_content["sentences"]  # Get split sentences from target

            for source_sentence in source_sentences:
                for target_sentence in target_sentences:
                    # Compute similarity for each sentence pair
                    source_vector = sentence_to_vector(source_sentence, model)
                    target_vector = sentence_to_vector(target_sentence, model)

                    similarity = cosine_similarity([source_vector], [target_vector])[0][0]

                    # Use an expander to display the results collapsibly
                    with st.expander(f"Compare: {source_sentence[:50]}... ↔ {target_sentence[:50]}..."):
                        st.markdown(
                            f"""
                            <p style="color:blue;">Source Sentence: <strong>{source_sentence}</strong></p>
                            <p style="color:green;">Target Sentence: <strong>{target_sentence}</strong></p>
                            <p style="color:orange;">Similarity: <strong>{similarity:.4f}</strong></p>
                            """,
                            unsafe_allow_html=True
                        )

    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Compute similarity matrix
    if sentences:
        similarity_matrix = compute_similarity_matrix(sentences, model)
        st.success("Similarity matrix successfully calculated.")

        # Convert similarity matrix to DataFrame for easier viewing
        similarity_df = pd.DataFrame(similarity_matrix, columns=[f"Sentence {i+1}" for i in range(len(sentences))],
                                    index=[f"Sentence {i+1}" for i in range(len(sentences))])

        # Display the similarity matrix as a table
        st.write("### Similarity Matrix")
        st.dataframe(similarity_df, width=800)  # You can adjust the width as needed

        # Optionally, display the similarity matrix as a heatmap
        st.write("### Similarity Matrix Heatmap")
        plt.figure(figsize=(10, 8))
        sns.heatmap(similarity_df, annot=True, cmap="YlGnBu", cbar=True, fmt=".4f", linewidths=0.5)
        st.pyplot(plt)
    else:
        st.error("No sentences found to compute the similarity matrix.")
