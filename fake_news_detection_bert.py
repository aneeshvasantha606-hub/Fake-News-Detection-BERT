# Fake News Detection using BERT (With Confusion Matrix)
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# 1. Load Dataset
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

# Labeling
fake["label"] = 0   # Fake
true["label"] = 1   # Real

data = pd.concat([fake, true]).reset_index(drop=True)
data = data[['text', 'label']].dropna()
print("✅ Data loaded successfully! Total samples:", len(data))

# 2. Tokenize text using BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_texts(texts, max_len=128):
    return tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )

# 3. Create Dataset Class
class NewsDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenize_texts(texts)
        self.labels = torch.tensor(labels.values, dtype=torch.long)
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item
    def __len__(self):
        return len(self.labels)
    
# 4. Split into Train/Test
train_texts, test_texts, y_train, y_test = train_test_split(
    data['text'], data['label'], test_size=0.2, random_state=42
)

train_dataset = NewsDataset(train_texts, y_train)
test_dataset = NewsDataset(test_texts, y_test)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# 5. Load Pretrained BERT
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bert = BertModel.from_pretrained('bert-base-uncased').to(device)
bert.eval()

# 6. Extract BERT Embeddings
def get_bert_embeddings(dataloader):
    embeddings = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting BERT embeddings"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = bert(input_ids=input_ids, attention_mask=attention_mask)
            # Take [CLS] token embedding
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu()
            embeddings.append(cls_embeddings)
    return torch.cat(embeddings)

X_train = get_bert_embeddings(train_loader)
X_test = get_bert_embeddings(test_loader)

# 7. Train Model (Logistic Regression)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
print("✅ Model trained successfully!")

# 8. Evaluate Model
y_pred = model.predict(X_test)

print("\n🎯 MODEL EVALUATION RESULTS 🎯")
print("Accuracy :", round(accuracy_score(y_test, y_pred), 3))
print("Precision:", round(precision_score(y_test, y_pred), 3))
print("Recall   :", round(recall_score(y_test, y_pred), 3))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\n🧩 Confusion Matrix:")
print(cm)

# Analyze False Positives and False Negatives
tn, fp, fn, tp = cm.ravel()
print(f"\n🔍 Analysis:")
print(f"True Positives (Real correctly identified): {tp}")
print(f"True Negatives (Fake correctly identified): {tn}")
print(f"False Positives (Fake misclassified as Real): {fp}")
print(f"False Negatives (Real misclassified as Fake): {fn}")

# 9. Simple User Input Prediction
def predict_news(text):
    inputs = tokenize_texts([text])
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        output = bert(**inputs)
        embedding = output.last_hidden_state[:, 0, :].cpu()
    pred = model.predict(embedding)
    return "REAL NEWS ✅" if pred[0] == 1 else "FAKE NEWS ❌"

while True:
    print("\nEnter news text (or type 'exit' to quit):")
    text = input("> ")
    if text.lower() == "exit":
        break
    print("Prediction:", predict_news(text))
