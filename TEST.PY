import re
import streamlit as st



label = None

st.set_page_config(layout ="wide")



# choisir le documents que lon veut 
# calculer la distance de ce documents avec everything 

# ou chercher : tu choisie 1 ou tout(merge all !)
# quoi chercher ? choisie un documents 
                                   





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


















# Define the raw texts for each corpus
corpus_texts = {
    'obama': """If there is anyone out there who still doubts that America is a place where all things are possible; who still wonders if the dream of our founders is alive in our time; who still questions the power of our democracy, tonight is your answer.
It's the answer told by lines that stretched around schools and churches in numbers this nation has never seen; by people who waited three hours and four hours, many for the very first time in their lives, because they believed that this time must be different; that their voice could be that difference.
It's the answer spoken by young and old, rich and poor, Democrat and Republican, black, white, Latino, Asian, Native American, gay, straight, disabled and not disabled  Americans who sent a message to the world that we have never been a collection of red states and blue states; we are, and always will be, the United States of America.
It's the answer that led those who have been told for so long by so many to be cynical, and fearful, and doubtful of what we can achieve to put their hands on the arc of history and bend it once more toward the hope of a better day.
It's been a long time coming, but tonight, because of what we did on this day, in this election, at this defining moment, change has come to America.
I just received a very gracious call from Sen. McCain. He fought long and hard in this campaign, and he's fought even longer and harder for the country he loves. He has endured sacrifices for America that most of us cannot begin to imagine, and we are better off for the service rendered by this brave and selfless leader. I congratulate him and Gov. Palin for all they have achieved, and I look forward to working with them to renew this nation's promise in the months ahead.
I want to thank my partner in this journey, a man who campaigned from his heart and spoke for the men and women he grew up with on the streets of Scranton and rode with on that train home to Delaware, the vice-president-elect of the United States, Joe Biden.
I would not be standing here tonight without the unyielding support of my best friend for the last 16 years, the rock of our family and the love of my life, our nation's next first lady, Michelle Obama. Sasha and Malia, I love you both so much, and you have earned the new puppy that's coming with us to the White House. And while she's no longer with us, I know my grandmother is watching, along with the family that made me who I am. I miss them tonight, and know that my debt to them is beyond measure.
To my campaign manager, David Plouffe; my chief strategist, David Axelrod; and the best campaign team ever assembled in the history of politics  you made this happen, and I am forever grateful for what you've sacrificed to get it done.
But above all, I will never forget who this victory truly belongs to  it belongs to you.
I was never the likeliest candidate for this office. We didn't start with much money or many endorsements. Our campaign was not hatched in the halls of Washington  it began in the backyards of Des Moines and the living rooms of Concord and the front porches of Charleston.
It was built by working men and women who dug into what little savings they had to give $5 and $10 and $20 to this cause. It grew strength from the young people who rejected the myth of their generation's apathy; who left their homes and their families for jobs that offered little pay and less sleep; from the not-so-young people who braved the bitter cold and scorching heat to knock on the doors of perfect strangers; from the millions of Americans who volunteered and organized, and proved that more than two centuries later, a government of the people, by the people and for the people has not perished from this earth. This is your victory.
I know you didn't do this just to win an election, and I know you didn't do it for me. You did it because you understand the enormity of the task that lies ahead. For even as we celebrate tonight, we know the challenges that tomorrow will bring are the greatest of our lifetime  two wars, a planet in peril, the worst financial crisis in a century. Even as we stand here tonight, we know there are brave Americans waking up in the deserts of Iraq and the mountains of Afghanistan to risk their lives for us. There are mothers and fathers who will lie awake after their children fall asleep and wonder how they'll make the mortgage, or pay their doctor's bills, or save enough for college. There is new energy to harness and new jobs to be created; new schools to build and threats to meet and alliances to repair.
The road ahead will be long. Our climb will be steep. We may not get there in one year, or even one term, but America  I have never been more hopeful than I am tonight that we will get there. I promise you: We as a people will get there.
There will be setbacks and false starts. There are many who won't agree with every decision or policy I make as president, and we know that government can't solve every problem. But I will always be honest with you about the challenges we face. I will listen to you, especially when we disagree. And, above all, I will ask you join in the work of remaking this nation the only way it's been done in America for 221 years  block by block, brick by brick, callused hand by callused hand.
What began 21 months ago in the depths of winter must not end on this autumn night. This victory alone is not the change we seek  it is only the chance for us to make that change. And that cannot happen if we go back to the way things were. It cannot happen without you.
So let us summon a new spirit of patriotism; of service and responsibility where each of us resolves to pitch in and work harder and look after not only ourselves, but each other. Let us remember that if this financial crisis taught us anything, it's that we cannot have a thriving Wall Street while Main Street suffers. In this country, we rise or fall as one nation  as one people.
Let us resist the temptation to fall back on the same partisanship and pettiness and immaturity that has poisoned our politics for so long. Let us remember that it was a man from this state who first carried the banner of the Republican Party to the White House  a party founded on the values of self-reliance, individual liberty and national unity. Those are values we all share, and while the Democratic Party has won a great victory tonight, we do so with a measure of humility and determination to heal the divides that have held back our progress.
As Lincoln said to a nation far more divided than ours, "We are not enemies, but friends... Though passion may have strained, it must not break our bonds of affection." And, to those Americans whose support I have yet to earn, I may not have won your vote, but I hear your voices, I need your help, and I will be your president, too.
And to all those watching tonight from beyond our shores, from parliaments and palaces to those who are huddled around radios in the forgotten corners of our world  our stories are singular, but our destiny is shared, and a new dawn of American leadership is at hand. To those who would tear this world down: We will defeat you. To those who seek peace and security: We support you. And to all those who have wondered if America's beacon still burns as bright: Tonight, we proved once more that the true strength of our nation comes not from the might of our arms or the scale of our wealth, but from the enduring power of our ideals: democracy, liberty, opportunity and unyielding hope.
For that is the true genius of America  that America can change. Our union can be perfected. And what we have already achieved gives us hope for what we can and must achieve tomorrow.
This election had many firsts and many stories that will be told for generations. But one that's on my mind tonight is about a woman who cast her ballot in Atlanta. She's a lot like the millions of others who stood in line to make their voice heard in this election, except for one thing: Ann Nixon Cooper is 106 years old.
She was born just a generation past slavery; a time when there were no cars on the road or planes in the sky; when someone like her couldn't vote for two reasons  because she was a woman and because of the color of her skin.
And tonight, I think about all that she's seen throughout her century in America  the heartache and the hope; the struggle and the progress; the times we were told that we can't and the people who pressed on with that American creed: Yes, we can.
At a time when women's voices were silenced and their hopes dismissed, she lived to see them stand up and speak out and reach for the ballot. Yes, we can.
When there was despair in the Dust Bowl and depression across the land, she saw a nation conquer fear itself with a New Deal, new jobs and a new sense of common purpose. Yes, we can.
When the bombs fell on our harbor and tyranny threatened the world, she was there to witness a generation rise to greatness and a democracy was saved. Yes, we can.
She was there for the buses in Montgomery, the hoses in Birmingham, a bridge in Selma and a preacher from Atlanta who told a people that "We Shall Overcome." Yes, we can.
A man touched down on the moon, a wall came down in Berlin, a world was connected by our own science and imagination. And this year, in this election, she touched her finger to a screen and cast her vote, because after 106 years in America, through the best of times and the darkest of hours, she knows how America can change. Yes, we can.
America, we have come so far. We have seen so much. But there is so much more to do. So tonight, let us ask ourselves: If our children should live to see the next century; if my daughters should be so lucky to live as long as Ann Nixon Cooper, what change will they see? What progress will we have made?
This is our chance to answer that call. This is our moment. This is our time  to put our people back to work and open doors of opportunity for our kids; to restore prosperity and promote the cause of peace; to reclaim the American Dream and reaffirm that fundamental truth that out of many, we are one; that while we breathe, we hope, and where we are met with cynicism, and doubt, and those who tell us that we can't, we will respond with that timeless creed that sums up the spirit of a people: Yes, we can.
Thank you, God bless you, and may God bless the United States of America."""
,'chiraq':  """
La confiance que vous venez de me témoigner, je veux y répondre en m'engageant dans l'action avec détermination.
Mes chers compatriotes de métropole, d'outre-mer et de l'étranger,
Nous venons de vivre un temps de grave inquiétude pour la Nation.
Mais ce soir, dans un grand élan la France a réaffirmé son attachement aux valeurs de la République.
Je salue la France, fidèle à elle-même, fidèle à ses grands idéaux, fidèle à sa vocation universelle et humaniste.
Je salue la France qui, comme toujours dans les moments difficiles, sait se retrouver sur l'essentiel. Je salue les Françaises et les Français épris de solidarité et de liberté, soucieux de s'ouvrir à l'Europe et au monde, tournés vers l'avenir.
J'ai entendu et compris votre appel pour que la République vive, pour que la Nation se rassemble, pour que la politique change. Tout dans l'action qui sera conduite, devra répondre à cet appel et s'inspirer d'une exigence de service et d'écoute pour chaque Française et chaque Français.
Ce soir, je veux vous dire aussi mon émotion et le sentiment que j'ai de la responsabilité qui m'incombe.
Votre choix d'aujourd'hui est un choix fondateur, un choix qui renouvelle notre pacte républicain. Ce choix m'oblige comme il oblige chaque responsable de notre pays. Chacun mesure bien, à l'aune de notre histoire, la force de ce moment exceptionnel.
Votre décision, vous l'avez prise en conscience, en dépassant les clivages traditionnels, et, pour certains d'entre vous, en allant au-delà même de vos préférences personnelles ou politiques.
La confiance que vous venez de me témoigner, je veux y répondre en m'engageant dans l'action avec détermination.
Président de tous les Français, je veux y répondre dans un esprit de rassemblement. Je veux mettre la République au service de tous. Je veux que les valeurs de liberté, d'égalité et de fraternité reprennent toute leur place dans la vie de chacune et de chacun d'entre nous.
La liberté, c'est la sécurité, la lutte contre la violence, le refus de l'impunité. Faire reculer l'insécurité est la première priorité de l'Etat pour les temps à venir.
La liberté, c'est aussi la reconnaissance du travail et du mérite, la réduction des charges et des impôts.
L'égalité, c'est le refus de toute discrimination, ce sont les mêmes droits et les mêmes devoirs pour tous.
La fraternité, c'est sauvegarder les retraites. C'est aider les familles à jouer pleinement leur rôle. C'est faire en sorte que personne n'éprouve plus le sentiment d'être laissé pour compte.
La France, forte de sa cohésion sociale et de son dynamisme économique, portera en Europe et dans le monde l'ambition de la paix, des libertés et de la solidarité.
Dans les prochains jours, je mettrai en place un gouvernement de mission, un gouvernement qui aura pour seule tâche de répondre à vos préoccupations et d'apporter des solutions à des problèmes trop longtemps négligés. Son premier devoir sera de rétablir l'autorité de l'Etat pour répondre à l'exigence de sécurité, et de mettre la France sur un nouveau chemin de croissance et d'emploi.
C'est par une action forte et déterminée, c'est par la solidarité de la Nation, c'est par l'efficacité des résultats obtenus, que nous pourrons lutter contre l'intolérance, faire reculer l'extrémisme, garantir la vitalité de notre démocratie. Cette exigence s'impose à chacun d'entre nous. Elle impliquera, au cours des prochaines années, vigilance et mobilisation de la part de tous.
Mes chers compatriotes,
Le mandat que vous m'avez confié, je l'exercerai dans un esprit d'ouverture et de concorde, avec pour exigence l'unité de la République, la cohésion de la Nation et le respect de l'autorité de l'Etat.
Les jours que nous venons de vivre ont ranimé la vigueur nationale, la vigueur de l'idéal démocratique français. Ils ont exprimé une autre idée de la politique, une autre idée de la citoyenneté.
Chacune et chacun d'entre vous, conscient de ses responsabilités, par un choix de liberté, a contribué, ce soir, à forger le destin de la France.
Il y a là un espoir qui ne demande qu'à grandir, un espoir que je veux servir.
Vive la République !
Vive la France !
""", 


}

# Add a selectbox to the sidebar for the corpus, including "Custom Text" option
import streamlit as st

# Existing code for selecting a corpus
selected_corpus = st.sidebar.selectbox(
    'Select a corpus',
    ('obama', 'chiraq', 'Custom Text', 'Upload Text File')  # Add 'Upload Text File' option
)


#choisir corpus 
#choisir un doc dans ce corpus 
#lister les document avec leur contenue
#calculer les distance avec tout les doc dans ce corpus + autres doc dans autres corpus (on cherche ou ?)


# Check if the user selected "Custom Text"
if selected_corpus == 'Custom Text':
    # Display a text area for user input
    custom_text = st.sidebar.text_area("Enter your custom text here", height=200)
    if custom_text:
        initial_text = custom_text
    else:
        initial_text = ""  # If no input is provided, set initial_text to an empty string

# Check if the user selected "Upload Text File"
elif selected_corpus == 'Upload Text File':
    uploaded_file = st.sidebar.file_uploader("Choose a text file", type="txt")
    if uploaded_file is not None:
        initial_text = uploaded_file.read().decode("utf-8")  # Read the file and decode to string
    else:
        initial_text = ""  # Set to empty string if no file is uploaded

else:
    # Get the raw text based on the selected corpus
    initial_text = corpus_texts[selected_corpus]


# Split the selected corpus text into sentences and tokens
if initial_text:
    sentences = split_into_sentences(initial_text)
    tokens = preprocess_text(sentences)
    token_count = len(tokens)
    sentences_no_punctuation_count = len(sentences)

    with st.expander("Corpus Information", expanded=True):  # 'expanded=True' pour qu'il soit ouvert par défaut
        st.markdown(
            f'<p style="color:orange;">This corpus contains {token_count} tokens and {sentences_no_punctuation_count} sentences</p>',
            unsafe_allow_html=True
        )
    # Dynamically populate the "Documents" selectbox with sentences as options
    sentence_options = ["Whole Corpus"] + [f"Document {i+1}: {sentence}" for i, sentence in enumerate(sentences)]

    # Display the document selectbox with a small orange text below the label
    selected_document = st.sidebar.selectbox('Documents', sentence_options)

    # Display the selected document with color (e.g., orange)

    # Check if the user selected "Whole Corpus"
    if selected_document == "Whole Corpus":
        selected_text = initial_text  # The whole text is selected
        idx = "Whole Corpus"
    else:
        # If the document is not the whole corpus, display the selected sentence
        selected_text = selected_document.split(": ", 1)[1]
        idx = selected_document.split(":")[0]
        sentences = split_into_sentences(selected_text)
        tokens = preprocess_text(sentences)
        sentences = tokens

    # Display the selected text in a text area for the user to view or edit
    user_input = st.sidebar.text_area(f"Chosen document: {idx}", value=selected_text, height=200)
else:
    st.markdown(f'<p style="color:red;">No text available. Please enter custom text or select a corpus.</p>', unsafe_allow_html=True)
















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






# After your existing code for corpus and document selection...

# Ensure the similarity matrix is calculated based on the choosen_matrix
if initial_text:  # Make sure there is text to work with
    # Calculate the similarity matrix using the chosen matrix (replace 'choosen_matrix' with your actual matrix)
    similarity_matrix = calculate_similarity_matrix(choosen_matrix)

    # Sidebar for K nearest documents
    st.sidebar.header("K Nearest Documents : ")

    # Slider for selecting K
    k = st.sidebar.slider("Select K", min_value=1, max_value=len(sentences), value=2)

    # Dropdown for selecting the document
    document_options = ["Whole Corpus"] + [f"Document {i + 1}: {sentences[i][:300]}" for i in range(len(sentences))]
    selected_document = st.sidebar.selectbox("Select Document", document_options)

    # Extract the document index based on the selection
    if selected_document == "Whole Corpus":
        doc_requete_index = 0  # Assuming the whole corpus is indexed as 0
    else:
        import re

        # Use regex to extract the document number
        match = re.search(r"Document (\d+):", selected_document)
        if match:
            doc_number = int(match.group(1)) - 1  # Convert to zero-based index
        else:
            doc_number = 0  # Default to 0 if "Whole Corpus" is selected
        # Get the document number
        doc_requete_index = doc_number  # Use document number directly if indexing starts from 1 for individual documents

    # Button to execute the function
# Button to execute the function
if st.sidebar.button("Find K Nearest Documents"):
    nearest_docs = k_plus_proches_documents(doc_requete_index, similarity_matrix, k)

    # Display results in an expander for better organization
    if nearest_docs:  # Check if there are results
        with st.expander(f"Les K plus proches Documents pour le Document sélectionné:", expanded=True):
            st.markdown(
                    f"<strong style='color:green;' >Document choisie : </strong> <strong style='color:orange;' >{selected_document } </strong>",
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





#DISPLAY SENTENCES : 


numbered_sentences = [f"{i+1}: {sentence.strip()}" for i, sentence in enumerate(sentences) if sentence.strip()]

# Display the list of sentences in a dropdown (liste déroulante)
selected_sentence = st.selectbox("Select a sentence", numbered_sentences)

# Display the selected sentence
st.markdown(f"**You selected:** {selected_sentence}")









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








# on implemente : ou chercher ? 
# on choisie ou chercher et on lajoute a nos distances

# sa peut etre un fichier ou plusieur fichier : 
# si plusieur fichier on les merge et puis on calcule