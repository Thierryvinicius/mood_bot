import os
import pandas as pd
import torch
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from model import Net
from preprocess import preprocess_text
from train import train_model, model_test
from inference import predict_sentiment
from config import INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, TEST_SPLIT, VAL_SPLIT, DATASET_PATH

# Carregar o dataset
if DATASET_PATH is None:
    dataset = pd.read_csv('https://raw.githubusercontent.com/futurexskill/ml-model-deployment/main/Restaurant_Reviews.tsv.txt', delimiter='\t')
    dataset.rename(columns={'Review': 'Phrase'}, inplace=True)
else:
    dataset = pd.read_csv(DATASET_PATH, delimiter='\t')
    print(dataset.head())
    print(dataset.columns)
    dataset.columns = ['Phrase', 'Note']
# Pré-processar o texto

corpus = preprocess_text(dataset)

# Vetorização TF-IDF
vectorizer = TfidfVectorizer(max_features=INPUT_SIZE, min_df=1, max_df=0.8)#voltar para 3 e 0.6 dps
X = vectorizer.fit_transform(corpus).toarray()
INPUT_SIZE = X.shape[1]
print(f'Número de características: {INPUT_SIZE}')
y = dataset.iloc[:, 1].values

# salvando o vectorizer
VECTOR_PATH = os.path.join(os.getcwd(),"models", "tfidf_vectorizer.pkl")
with open(VECTOR_PATH, 'wb') as f:
    pickle.dump(vectorizer, f)
print('VETOR TF-IDF SALVO EM, ', VECTOR_PATH)

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SPLIT, random_state=0)

#dividir os dados do treino em treino e validação
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=VAL_SPLIT, random_state=0) #25% dos dados de treino que é 20% do total


# Converter para tensores
X_train_tensor = torch.from_numpy(X_train).float()
y_train_tensor = torch.from_numpy(y_train).long()
X_val_tensor = torch.from_numpy(X_val).float()
y_val_tensor = torch.from_numpy(y_val).long()
X_test_tensor = torch.from_numpy(X_test).float()
y_test_tensor = torch.from_numpy(y_test).long()


model = Net(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
train_model(model, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor)
model_test(model, X_test_tensor, y_test_tensor)

# Testar o modelo com novas amostras
sample1 = "Great match from Stephen Curry"
sample2 = "bad performance by Serbia in the basketball match"

print(predict_sentiment(model, vectorizer, sample1))
print(predict_sentiment(model, vectorizer, sample2))

# Salvar o modelo e o vetor TF-IDF
#torch.save(model.state_dict(), 'text_classifier_pytorch.pth')
