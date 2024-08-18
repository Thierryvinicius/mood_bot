
import torch

def predict_sentiment(model, vectorizer, review):
    model.eval()
    review_processed = vectorizer.transform([review]).toarray()
    review_tensor = torch.from_numpy(review_processed).float()
    with torch.no_grad():
        output = model(review_tensor)
        sentiment = torch.argmax(output, dim=1).item()
        return "Positive" if sentiment == 1 else "Negative"
