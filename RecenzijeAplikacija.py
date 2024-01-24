import pandas as pd
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import seaborn as sns
import matplotlib.pyplot as plt
import os.path

def napravi_recenziju():
    # Učitavanje modela i vektorizatora
    with open('model_Restaurant.pkl', 'rb') as datoteka:
        model = pickle.load(datoteka)
    with open('vektorizator_Restaurant.pkl', 'rb') as datoteka:
        vektorizator = pickle.load(datoteka)

    # Unos nove recenzije
    nova_recenzija = input("Unesite novu recenziju: ")

    # Predobrada teksta
    zaustavne_rijeci = set(stopwords.words('english'))
    nova_recenzija = ' '.join([rijec for rijec in word_tokenize(nova_recenzija) if rijec not in zaustavne_rijeci])
    # Vektorizacija recenzije
    X_nova = vektorizator.transform([nova_recenzija])
    
    # Predviđanje ocjene
    predikcija = model.predict(X_nova)
    #ako je ocjena 1 ili 2, recenzija je "kritika" inače je "pohvala"
    if predikcija[0] == 1 or predikcija[0] == 2:
        informacija = "(Kritika)"
    elif predikcija[0] == 3:
        informacija = "(Neutralna)"
    else:
        informacija = "(Pohvala)"

    # Ispis ocjene
    print("Ocjena recenzije je:", predikcija[0], informacija)

    # Spremanje recenzije i ocjene u CSV datoteku
    df_nova = pd.DataFrame({'Review': [nova_recenzija], 'Rating': [predikcija[0]]})
    if not os.path.isfile('nove_recenzije.csv'):
        df_nova.to_csv('nove_recenzije.csv', index=False)
    else: # inače se dodaje na postojeću datoteku
        df_nova.to_csv('nove_recenzije.csv', mode='a', header=False, index=False)

def prikaz_grafa():
    try:
        df_nova = pd.read_csv('nove_recenzije.csv')
        sns.countplot(x='Rating', data=df_nova)
        plt.xlabel('Ocjena recenzije')
        plt.ylabel('Broj recenzija')
        plt.title('Broj recenzija po ocjenama')
        plt.show()
    except:
         print("Nema recenzija!")

# Switch case
option = input("Odaberite: 1) Napravi novu recenziju, 2) Prikaži podatke : ")

if option == "1":
    napravi_recenziju()
elif option == "2":
    prikaz_grafa()
else:
    print("Nepostojeća opcija!")