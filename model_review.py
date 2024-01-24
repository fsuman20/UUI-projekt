# Učitavanje potrebnih biblioteka
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from imblearn.over_sampling import RandomOverSampler
from sklearn.naive_bayes import MultinomialNB

# Učitavanje podataka
df = pd.read_csv('Restaurant reviews.csv')

# Transformacija i čišćenje podataka
df['Rating'] = df['Rating'].replace({'Like':3})
df['Rating'] = df['Rating'].fillna(df['Rating'].median())
df['Rating'] = df['Rating'].astype(float).round().astype(int)

# Eksplorativna analiza podataka
print(df.info())
print(df.describe())

# Vizualizacija podataka
sns.countplot(x='Rating', data=df)
plt.xlabel('Ocjena recenzije')
plt.ylabel('Broj recenzija')
plt.title('Broj recenzija po ocjenama')
plt.show()

# Tokenizacija i vektorizacija
zaustavne_rijeci = set(stopwords.words('english'))
df['Review'] = df['Review'].astype(str).apply(lambda x: ' '.join([rijec for rijec in word_tokenize(x) if rijec not in zaustavne_rijeci]))

vektorizator = TfidfVectorizer()
X = vektorizator.fit_transform(df['Review'])

# Balansiranje klasa
ros = RandomOverSampler(random_state=0)
X_resampled, y_resampled = ros.fit_resample(X, df['Rating'])

# Podjela podataka na trening i test set
X_trening, X_test, y_trening, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Definiranje parametara za pretragu
parametri = {'alpha': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}

# Inicijalizacija modela
model = MultinomialNB()

# Inicijalizacija GridSearchCV
grid_search = GridSearchCV(model, parametri, cv=5, scoring='accuracy')

# Treniranje modela koristeći GridSearchCV
grid_search.fit(X_trening, y_trening)

# Ispis najboljih parametara
print('Najbolji parametri:', grid_search.best_params_)

model = MultinomialNB(alpha=0.1)
model.fit(X_trening, y_trening)

# Evaluacija modela
y_predikcija = model.predict(X_test)
print('Točnost modela:', accuracy_score(y_test, y_predikcija))
print('Matrica konfuzije:\n', confusion_matrix(y_test, y_predikcija))

# Vizualizacija matrice konfuzije pomoću ConfusionMatrixDisplay
cm = confusion_matrix(y_test, y_predikcija)
cmd = ConfusionMatrixDisplay(cm, display_labels=[1,2,3,4,5])
cmd.plot()
plt.xlabel('Predviđena ocjena')
plt.ylabel('Stvarna ocjena')
plt.title('Matrica konfuzije')
plt.show()

# Unakrsna validacija
ocjene = cross_val_score(model, X_resampled, y_resampled, cv=5)
print('Unakrsne ocjene:', ocjene)
print('Prosječna ocjena:', ocjene.mean())

# Spremanje modela
with open('model_Restaurant.pkl', 'wb') as datoteka:
    pickle.dump(model, datoteka)

# Spremanje vektorizatora
with open('vektorizator_Restaurant.pkl', 'wb') as datoteka:
    pickle.dump(vektorizator, datoteka)