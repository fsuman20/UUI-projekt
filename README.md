# Procesiranje prirodnog jezika za razumijevanje povratnih informacija klijenata u ugostiteljskoj industriji
## Projekt iz kolegija uvod u umjetnu inteligenciju

Ovaj repozitorij sadrži izvorni kod i dokumentaciju za seminarski rad iz kolegija Uvod u umjetnu inteligenciju na Fakultetu organizacije i informatike.

## Opis projekta
Projekt se bavi analizom recenzija restorana koristeći razne tehnike procesiranja prirodnog jezika (NLP). Cilj je napraviti aplikaciju koja može predvidjeti ocjenu restorana na temelju teksta recenzije, te vizualizirati rezultate i statistike. Projekt se sastoji od dva dijela: modela i aplikacije.

Model je izrađen u Python programu koji koristi statističko strojno učenje i NLP tehnike za treniranje i evaluaciju klasifikatora recenzija. Model koristi skup podataka od 10000 recenzija restorana preuzetih s Kaggle repozitorija. Temelji na naivnom Bayesovom klasifikatoru, te koristi TF-IDF vektorizaciju, balansiranje klasa, pretraživanje po rešetci i unakrsnu validaciju. Model i vektorizator se spremaju u datoteke za daljnju upotrebu u aplikaciji.

Aplikacija za recenziranje je izrađena u Python programu koji omogućuje korisniku da unese novu recenziju restorana i dobije predviđenu ocjenu na temelju modela. Aplikacija također prikazuje grafički prikaz broja recenzija po ocjenama, te omogućuje korisniku da odabere između stvaranja nove recenzije i prikaza podataka.

## Upute za pokretanje
Za pokretanje projekta potrebno je imati instaliran Python (poželjno najnoviji javni), te sljedeće biblioteke: pandas, pickle, matplotlib, seaborn, sklearn, nltk i imblearn. Poželjno je imati Conda okruženje sa svim uključenim bibliotekama. Također je potrebno preuzeti skup podataka `Restaurant reviews.csv` s repozitorija i spremiti ga u isti direktorij kao i Python programe.

Za pokretanje modela, potrebno je pokrenuti program `model_review.py`. Program će učitati podatke, očistiti ih, vektorizirati ih, balansirati ih, podijeliti ih na trening i test set, optimizirati parametre modela, trenirati model, evaluirati model, vizualizirati podatke i spremiti model i vektorizator u datoteke.

Za pokretanje aplikacije, potrebno je pokrenuti program `RecenzijeAplikacija.py`. Program će učitati model i vektorizator iz datoteka, te omogućiti korisniku da odabere opciju: 

1) Napravi novu recenziju
2) Prikaži podatke

Ako korisnik odabere prvu opciju, program će zatražiti unos nove recenzije, očistiti je, vektorizirati je, predvidjeti ocjenu, ispisati ocjenu, te spremiti recenziju i ocjenu u CSV datoteku ‘nove_recenzije.csv’. Ako korisnik odabere drugu opciju, program će prikazati grafički prikaz broja recenzija po ocjenama za nove recenzije.

## Autor
Frane Suman, student Informacijskih i poslovnih sustava na Fakultetu organizacije i informatike. Mentor: izv. prof. dr. sc. Dijana Oreški.
