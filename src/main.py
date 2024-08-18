
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from model import Net
from preprocess.py import preprocess_text
from train import train_model
from inference import predict_sentiment
from config import INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE

# Carregar o dataset
dataset = pd.read_csv('https://raw.githubusercontent.com/futurexskill/ml-model-deployment/main/Restaurant_Reviews.tsv.txt', delimiter='\t', quoting=3)

# Pré-processar o texto
corpus = preprocess_text(dataset)

# Vetorização TF-IDF
vectorizer = TfidfVectorizer(max_features=INPUT_SIZE, min_df=3, max_df=0.6)
X = vectorizer.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Converter para tensores
X_train_tensor = torch.from_numpy(X_train).float()
X_test_tensor = torch.from_numpy(X_test).float()
y_train_tensor = torch.from_numpy(y_train).long()

model = Net(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
train_model(model, X_train_tensor, y_train_tensor)

# Testar o modelo com novas amostras
sample1 = "Great match from Stephen Curry"
sample2 = "bad performance by Serbia in the basketball match"

print(predict_sentiment(model, vectorizer, sample1))
print(predict_sentiment(model, vectorizer, sample2))

# Salvar o modelo e o vetor TF-IDF
torch.save(model.state_dict(), 'text_classifier_pytorch.pth')

import pickle
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
