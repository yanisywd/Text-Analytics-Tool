# -*- coding: utf-8 -*-
"""Nuage_de_mots.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1lRYqr5aBoMKdQU3egQdurYuzMQe4sq1m

# **Création de nuages de mots**

# Mes imports
"""

from wordcloud import WordCloud , ImageColorGenerator
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image  # , ImageDraw, ImageFont
from random import randint
import nltk
from nltk.corpus import stopwords

nltk.download('punkt_tab')

"""# Initialisation des chaînes des caractères, vous pouvez lire le contenu à partir du(es) fichier(s)"""

Dis_Obama ="""If there is anyone out there  America."""

Dis_Chirac =""" La confiance que vous venez de me témoigner, je veux y répondre en m'engageant dans l'action avec détermination.

Vive la France ! """

# Vous pouvez convertir le texte en minuscule et retester les codes
#Dis_Obama = Dis_Obama.lower()
#Dis_Chirac = Dis_Chirac.lower()

# Vous pouvez aussi faire d'autres prétraitements et retester

"""# Affichage du nuage de mots avec les paramètres par défaut




"""

wc= WordCloud()
wc.generate(Dis_Obama)
plt.imshow(wc)
plt.show()

wc.generate(Dis_Chirac)
plt.imshow(wc)
plt.show()

"""# Suppression des axes"""

wc_Obama = WordCloud()
wc_Obama.generate(Dis_Obama)
plt.imshow(wc_Obama)
plt.axis("off")
plt.show()

wc_Chirac = WordCloud().generate(Dis_Chirac)
plt.imshow(wc_Chirac)
plt.axis("off")
plt.show()
# On remarque que les deux mots les plus fréquents sont "de" and "la"

"""## Les paramètres par défaut utilisés par WordCloud

*   width (par défaut : 400)
*   height (par défaut : 200)
*   margin (par défaut : 2)
*   prefer_horizontal (par défaut : 0.90)
*   mask (par défaut : None)
*   contour_color (par défaut : "black")
*   contour_width (par défaut : 0)
*   scale (par défaut : 1)   
*   max_words (par défaut : 200)  
*   etc.

Pour voir tous les paramètres avec leur valeur par défaut, vous pouvez utiliser _ _dict_ _ ou afficher directement la valeur d'un paramètre donné.

"""

wc= WordCloud()
print(wc.__dict__)

"""## Pour afficher juste les valeurs de quelques paramètres spécifiques, on accède aux attributs de la classe."""

print("Largeur :", wc.width)
print("Hauteur :", wc.height)
print("Couleur d'arrière-plan :", wc.background_color)
print("Nombre max de mots :", wc.max_words)

"""# Le paramètre stopwords:
On ramque que pour le texte en français, on a obtenu un nuage qui mets en évidence des mots vides mais ce n'est pas le cas pour le texte en anglais.

"""

print("Liste des mots vides utilisée par défaut :\n", wc.stopwords)

from wordcloud import STOPWORDS
print(STOPWORDS)
# donc la valeur par défaut du paramètre stopwords est la liste STOPWORDS du module STOPWORDS

"""# Utilisation de liste vide pour les stopwords pour vérification"""

wc_Obama = WordCloud(stopwords=[])
wc_Obama.generate(Dis_Obama)
plt.imshow(wc_Obama)
plt.axis("off")
plt.show()
# On remarque que les deux mots les plus fréquents sont "the" and "and"

"""# Personalisation des paramètres
Voici quelques exemples de paramètres:

* colormap: La carte de couleurs
* width: La largeur de l'image
* height: La hauteur de l'image
* background_color: la couleur du fond
* max_words: le nombre maximum de mots uniques utilisés
* stopwords: la liste de mots vides à exclure lors de l'affichage

Pour la liste complète, vous pouvez consulter cette page:

https://amueller.github.io/word_cloud/generated/wordcloud.WordCloud.html

"""

wc_Obama = WordCloud(colormap = 'binary',width=200, height=400,background_color = 'white' , max_words = 300)
wc_Obama.generate(Dis_Obama)
plt.imshow(wc_Obama)
plt.axis("off")
plt.show()

wc_Chirac = WordCloud(colormap = 'Spectral',width=400, height=400,background_color = 'white' , max_words = 100).generate(Dis_Chirac)
plt.imshow(wc_Chirac)
plt.axis("off")
plt.show()

"""# Suppression des mots vides avec stopwords

## Suppression des quelques mots vides de votre choix
*   Par exemple, on retire les deux mots les plus fréquents pour voir le nouveau résultat
"""

wc_Obama = WordCloud(stopwords=['the','and'] , colormap = 'binary',width=600, height=300,background_color = 'white' , max_words = 300)
wc_Obama.generate(Dis_Obama)
plt.imshow(wc_Obama)
plt.axis("off")
plt.show()

wc_Chirac = WordCloud(stopwords=['de','la'] , colormap = 'Spectral',width=400, height=400,background_color = 'white' , max_words = 100).generate(Dis_Chirac)
plt.imshow(wc_Chirac)
plt.axis("off")
plt.show()

"""## Suppression des mots vides en utilisant des listes pré-établies
*   Par exemple, par exemple on utilise la liste du module WordCloud ou nltk
"""

wc_Obama = WordCloud(stopwords=STOPWORDS , colormap = 'CMRmap',width=600, height=300,background_color = 'white' )
wc_Obama.generate(Dis_Obama)
plt.imshow(wc_Obama)
plt.axis("off")
plt.show()

"""## Utiliser la liste des stopwords de nltk pour voir la différence"""

nltk.download('stopwords')
english_stopwords = stopwords.words('english')

wc_Obama = WordCloud(stopwords=english_stopwords , colormap = 'CMRmap',width=600, height=300,background_color = 'white' )
wc_Obama.generate(Dis_Obama)
plt.imshow(wc_Obama)
plt.axis("off")
plt.show()

french_stopwords = stopwords.words('french')

wc_Chirac = WordCloud(stopwords=french_stopwords , colormap = 'Spectral',width=400, height=400,background_color = 'white' , max_words = 100).generate(Dis_Chirac)
plt.imshow(wc_Chirac)
plt.axis("off")
plt.show()

"""## Plusieurs cartes de couleurs sont disponibles. Vous pouvez les tester et choisir celle qui vous convient :
 https://matplotlib.org/2.0.0/examples/color/colormaps_reference.html



"""

wc_Obama = WordCloud(stopwords=english_stopwords , colormap = 'seismic' )
wc_Obama.generate(Dis_Obama)
plt.imshow(wc_Obama)
plt.axis("off")
plt.show()

wc_Chirac = WordCloud(stopwords=french_stopwords , colormap = 'Accent').generate(Dis_Chirac)
plt.imshow(wc_Chirac)
plt.axis("off")
plt.show()

"""#Utilisation d'un masque
Les mots ne seront pas affichés dans les zones où les valeurs de pixels sont égales à 255.
Vous pouvez manipuler votre masque pour définir vous même les zones de non écriture.

*   Exemple: **mask[mask == 1] = 255**

=> ici tous les pixels initialement à 1 seront remplacés par 255 pour éviter l'écriture dessus.
"""

mask_usa_map = np.array(Image.open('usa-map.jpg'))
wc_Obama =  WordCloud(mask=mask_usa_map).generate(Dis_Obama)
plt.axis("off")
plt.imshow(wc_Obama)
plt.show()

mask_france_map = np.array(Image.open('france-map.jpg'))
wc_Chirac =  WordCloud(stopwords=french_stopwords ,mask=mask_france_map).generate(Dis_Chirac)
plt.axis("off")
plt.imshow(wc_Chirac)
plt.show()

"""## Amérioration de l'affichage avec des paramètres de la forme du nuage (couleur de fond, de contour, etc)"""

wc_Obama =  WordCloud(background_color="white",mask=mask_usa_map, contour_width=2, contour_color='firebrick').generate(Dis_Obama)
plt.axis("off")
plt.imshow(wc_Obama)
plt.show()

wc_Chirac =  WordCloud(stopwords=french_stopwords+['C\'est','a','j\'ai','aussi','d\'autre'] ,background_color="white",mask=mask_france_map, contour_width=2, contour_color='firebrick').generate(Dis_Chirac)
plt.axis("off")
plt.imshow(wc_Chirac)
plt.show()

"""## Modification des couleurs d'affichage avec la méthode **recolor**

Une couleur peut être définie avec
* Un nom (darkgreen) : https://matplotlib.org/2.0.0/examples/color/named_colors.html
* Sous forme héxadécimale #Nuance_rouge_exNuance_vert_exNuance_bleu_ex (#ffbb11) : https://matplotlib.org/stable/users/explain/colors/colors.html
* Sous la forme rgb(nuance_rouge, nuance_vert, nuance_bleu) (rgb(255,45,97)): https://www.rapidtables.com/web/color/RGB_Color.html





"""

# Vous pouvez lancer ce code plusieurs fois pour voir les différences de couleurs
# composer une couleur aléatoire
def couleur(*args, **kwargs):
    return "rgb({}, {}, {})".format(randint(100, 255),randint(100, 255),randint(100, 255))

wordcloud = WordCloud(background_color="white",mask=mask_usa_map, contour_width=2, contour_color='firebrick').generate(Dis_Obama)
# Utiliser la couleur aléatoire pour afficher les mots
plt.imshow(wordcloud.recolor(color_func = couleur))
plt.axis("off")
plt.show()

# Vous pouvez lancer ce code plusieurs fois pour voir les différences de couleurs
# composer une couleur aléatoire
def couleur(*args, **kwargs):
    return "rgb({}, {}, {})".format(randint(100, 255),randint(100, 255),randint(100, 255))

wordcloud = WordCloud(stopwords=french_stopwords , background_color="white",mask=mask_france_map, contour_width=2, contour_color='firebrick').generate(Dis_Chirac)
# Utiliser la couleur aléatoire pour afficher les mots
plt.imshow(wordcloud.recolor(color_func = couleur))
plt.axis("off")
plt.show()

"""# Choix des couleurs d'affichage à partir des couleurs de l'image"""

# Generer un nuage de mots
mask_usa_flag = np.array(Image.open("flag-usa.png"))
wordcloud_usa = WordCloud(background_color="white",max_words=1000,mask=mask_usa_flag).generate(Dis_Obama)

# Définir les couleurs à partir de l'image du mask
image_colors = ImageColorGenerator(mask_usa_flag)
plt.figure(figsize=[7,7])

# Modifier les couleurs du nuage de mots selon celles du mask
plt.imshow(wordcloud_usa.recolor(color_func=image_colors), interpolation="bilinear")
plt.axis("off")
plt.show()

# Generer un nuage de mots
mask_france_flag = np.array(Image.open("flag-france.jpg"))
wordcloud_france = WordCloud(stopwords=french_stopwords+['C\'est','a','j\'ai','aussi','d\'autre'] ,background_color="white",max_words=1000, mask=mask_france_flag).generate(Dis_Chirac)

# Définir les couleurs à partir de l'image du mask
image_colors = ImageColorGenerator(mask_france_flag)
plt.figure(figsize=[7,7])

# Modifier les couleurs du nuage de mots selon celles du mask
plt.imshow(wordcloud_france.recolor(color_func=image_colors), interpolation="bilinear")
plt.axis("off")
plt.show()

"""# Sauvegarde des images des nuages de points"""

wc_Obama.to_file("word_map_usa.png")
wc_Chirac.to_file("word_map_france.png")
wordcloud_usa.to_file("word_flag_usa.png")
wordcloud_france.to_file("word_flag_france.png")

"""# La taille des mots dans le nuage de mots dépend de l'importance de chaque mot dans le texte. Par défaut l'improtance d'un mot est calculée selon son nombre d'occurences dans le texte.

## Vérification des nombres d'occurrences utiliés par WordCloud
"""

# Récupérer le dictionnaire des fréquences
Nb_occ_Dis_Obama = wc_Obama.process_text(Dis_Obama)
# Trier le dictionnaire d'une façon décroissante en fontion des valeurs des occurrences
print(sorted(Nb_occ_Dis_Obama.items(), key=lambda x: x[1], reverse=True))

# Récupérer le dictionnaire des fréquences
Nb_occ_Dis_Chirac = wc_Chirac.process_text(Dis_Chirac)
# Trier le dictionnaire d'une façon décroissante en fontion des valeurs des occurrences
print(sorted(Nb_occ_Dis_Chirac.items(), key=lambda x: x[1], reverse=True))

"""## Vérification des nombres des frequences utiliées par WordCloud
La taille de chaque mot est proportionnelle à sa fréquence relative par rapport au mot le plus fréquent. Le mot ayant la fréquence maximale sera affiché avec la plus grande taille possible
"""

# Récupérer le dictionnaire des fréquences
frequences_Dis_Obama = wc_Obama.words_
frequences_Dis_Chirac = wc_Chirac.words_

print(frequences_Dis_Obama)
print(frequences_Dis_Chirac)

"""# Calcul de l'importance des mots en fonction du score TFIDF"""

from nltk.tokenize import sent_tokenize

# Considérer le discours comme étant un corpus composé de documents (phrases)
corpus_Obama = sent_tokenize(Dis_Obama)
corpus_Chirac = sent_tokenize(Dis_Chirac)

from sklearn.feature_extraction.text import TfidfVectorizer

# Dans cet exemple, la liste des mots vides est passée au constructeur TFifdVectorizer
# C'est mieux de faire un pré-traitement comme on l'avait déjà fait auparavant.
# En éliminant les mots vides après la tokenisation et avant le calcul du tfidf

mots_vides =french_stopwords+['C\'est','a','j\'ai','aussi','d\'autre']

# Instantiation de deux objets de la classe TfidfVectorize
vectorizer_Obama = TfidfVectorizer(stop_words='english')
vectorizer_Chirac = TfidfVectorizer(stop_words = mots_vides)

# Récupérer la matrice TFIDF ainsi que le liste des mots du vocabulaire à partir du discours Obama
matrice_tfidf_Obama = vectorizer_Obama.fit_transform(corpus_Obama)
les_tokens_Obama = vectorizer_Obama.get_feature_names_out()

# Récupérer la matrice TFIDF ainsi que le liste des mots du vocabulaire à partir du discours Chirac
matrice_tfidf_Chirac = vectorizer_Chirac.fit_transform(corpus_Chirac)
les_tokens_Chirac = vectorizer_Chirac.get_feature_names_out()


# convertir les matrices en tableaux numpy
score_tfidf_Obama = matrice_tfidf_Obama.toarray()
score_tfidf_Chirac = matrice_tfidf_Chirac.toarray()

"""## Le score de chaque mot peut être calculé de différentes façons à partir de ses scores Tf-IDF dans les différentes phrases.

*   Calcul la somme des scores TF-IDF d'un mot à travers des différents documetns
*   Calcul la moyenne des scores TF-IDF d'un mot à travers des différents documetns
*    Calcul le maximum des scores TF-IDF d'un mot à travers des différents documetns
*   etc,

## L'importance du mot comme la somme de ses TFIDF à travers l'ensemble des documents
"""

score_tfidf_par_token_Obama = dict(zip(les_tokens_Obama, score_tfidf_Obama.sum(axis=0)))

score_tfidf_par_token_Chirac = dict(zip(les_tokens_Chirac, score_tfidf_Chirac.sum(axis=0)))


print(sorted(score_tfidf_par_token_Obama.items(), key=lambda x: x[1], reverse=True))
print(sorted(score_tfidf_par_token_Chirac.items(), key=lambda x: x[1], reverse=True))

"""### Affichage des nuages de mots en utilisant la somme des TFIDFs"""

wc_Obama = WordCloud(background_color="white",mask=mask_usa_map, contour_width=2, contour_color='firebrick')
wc_Obama.generate_from_frequencies(score_tfidf_par_token_Obama)

# Afficher le nuage de mots
plt.imshow(wc_Obama, interpolation="bilinear")
plt.axis("off")
plt.show()

wc_Chirac = WordCloud(background_color="white",mask=mask_france_map, contour_width=2, contour_color='firebrick')
wc_Chirac.generate_from_frequencies(score_tfidf_par_token_Chirac)

# Afficher le nuage de mots
plt.imshow(wc_Chirac, interpolation="bilinear")
plt.axis("off")
plt.show()

"""### Affichage des nuages de mots en utilisant la moyenne des TFIDFs; l'ordre d'importance reste identique à celui de la somme"""

score_tfidf_par_token_Obama = dict(zip(les_tokens_Obama, score_tfidf_Obama.mean(axis=0)))

score_tfidf_par_token_Chirac = dict(zip(les_tokens_Chirac, score_tfidf_Chirac.mean(axis=0)))


print(sorted(score_tfidf_par_token_Obama.items(), key=lambda x: x[1], reverse=True))
print(sorted(score_tfidf_par_token_Chirac.items(), key=lambda x: x[1], reverse=True))

wc_Obama = WordCloud(background_color="white",mask=mask_usa_map, contour_width=2, contour_color='firebrick')
wc_Obama.generate_from_frequencies(score_tfidf_par_token_Obama)

# Afficher le nuage de mots
plt.imshow(wc_Obama, interpolation="bilinear")
plt.axis("off")
plt.show()

wc_Chirac = WordCloud(background_color="white",mask=mask_france_map, contour_width=2, contour_color='firebrick')
wc_Chirac.generate_from_frequencies(score_tfidf_par_token_Chirac)

# Afficher le nuage de mots
plt.imshow(wc_Chirac, interpolation="bilinear")
plt.axis("off")
plt.show()

"""### Affichage des nuages de mots en utilisant le maximum des TFIDFs


"""

score_tfidf_par_token_Obama = dict(zip(les_tokens_Obama, score_tfidf_Obama.max(axis=0)))

score_tfidf_par_token_Chirac = dict(zip(les_tokens_Chirac, score_tfidf_Chirac.max(axis=0)))


print(sorted(score_tfidf_par_token_Obama.items(), key=lambda x: x[1], reverse=True))
print(sorted(score_tfidf_par_token_Chirac.items(), key=lambda x: x[1], reverse=True))

wc_Obama = WordCloud(background_color="white",mask=mask_usa_map, contour_width=2, contour_color='firebrick')
wc_Obama.generate_from_frequencies(score_tfidf_par_token_Obama)

# Afficher le nuage de mots
plt.imshow(wc_Obama, interpolation="bilinear")
plt.axis("off")
plt.show()

wc_Chirac = WordCloud(background_color="white",mask=mask_france_map, contour_width=2, contour_color='firebrick')
wc_Chirac.generate_from_frequencies(score_tfidf_par_token_Chirac)

# Afficher le nuage de mots
plt.imshow(wc_Chirac, interpolation="bilinear")
plt.axis("off")
plt.show()