import streamlit as st
import os
import logging
import openai

def llmchatbot():
    # Set your OpenAI API key
    openai.api_key = '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'

    # Fonction pour extraire le texte des dossiers
    def extract_text_from_folders(folders):
        texts = []
        for folder in folders:
            if os.path.isfile(folder):  # If it's a file, read it directly
                if folder.endswith(".txt"):
                    try:
                        with open(folder, "r", encoding="utf-8") as f:
                            texts.append(f.read())
                    except Exception as e:
                        logging.error(f"Erreur lors de la lecture du fichier {folder}: {e}")
            elif os.path.isdir(folder):  # If it's a directory, walk through it
                for root, _, files in os.walk(folder):
                    for file in files:
                        if file.endswith(".txt"):
                            file_path = os.path.join(root, file)
                            try:
                                with open(file_path, "r", encoding="utf-8") as f:
                                    texts.append(f.read())
                            except Exception as e:
                                logging.error(f"Erreur lors de la lecture du fichier {file_path}: {e}")
            else:
                logging.warning(f"{folder} n'est ni un fichier texte ni un dossier valide.")
        return texts

    # Application Streamlit
    st.title("LLM Chatbot")

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
        

    # Entrée de la question par l'utilisateur
    question = st.text_input("Posez une question :")
    if question:
        # New API call format
        response = openai.ChatCompletion.create(
            model="gpt-4",  # Or any other model of your choice
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Based on the following text, answer the question or tell information that relate to the question respond in the correct language: '{question}'\n\nText: {raw_texts[0]}"}
            ],
            max_tokens=150  # Adjust the number of tokens according to your needs
        )

        # Extract the answer from the response
        answer = response['choices'][0]['message']['content']

        # Afficher les résultats
        st.markdown("### Réponse")
        st.write(f"**Réponse :** {answer}")

# Exécuter l'application Streamlit
