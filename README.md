# 📰 Fake News Detection using BERT (with Confusion Matrix)

This project implements a **Fake News Detection System** using **BERT embeddings** and a **Logistic Regression classifier**.  
It evaluates model performance with metrics such as **accuracy**, **precision**, **recall**, and a **confusion matrix**.  
You can also input your own news text to check whether it’s **real or fake** in real time.

---

## 🚀 Features
- Uses **BERT (bert-base-uncased)** for sentence embeddings  
- Trains a **Logistic Regression** model on BERT features  
- Displays **Accuracy**, **Precision**, **Recall**, and **Confusion Matrix**  
- Provides **interactive news prediction** via user input  
- Includes **False Positive / False Negative analysis**

---

## 🧠 Model Architecture
1. **Text Input**
2. **BERT Tokenization & Embedding**
3. **Feature Extraction** (CLS token representation)
4. **Logistic Regression Classifier**
5. **Evaluation Metrics + Confusion Matrix**

---

## 📦 Requirements

Install dependencies using pip:

```bash
pip install pandas torch scikit-learn transformers tqdm
```

Make sure you have **Python 3.8+** installed.

---

## 📁 Dataset
This project uses two CSV files:
- `Fake.csv` → contains fake news samples
- `True.csv` → contains real news samples

Each file must have a column named **`text`**.

Example:
```csv
text,label
"Breaking: Government denies rumors...",0
"New vaccine rollout begins today...",1
```

---

## ⚙️ How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/Fake-News-Detection-BERT.git
   cd Fake-News-Detection-BERT
   ```

2. Add the dataset files:
   - `Fake.csv`
   - `True.csv`

3. Run the script:
   ```bash
   python fake_news_detection_bert.py
   ```

4. Once the model is trained, enter news text manually:
   ```
   Enter news text (or type 'exit' to quit'):
   > The president announced a new policy today.
   Prediction: REAL NEWS ✅
   ```

---

## 📊 Example Confusion Matrix Output
```
🧩 Confusion Matrix:
[[935  72]
 [ 85 921]]

🔍 Analysis:
True Positives (Real correctly identified): 921
True Negatives (Fake correctly identified): 935
False Positives (Fake misclassified as Real): 72
False Negatives (Real misclassified as Fake): 85
```

---

## 🧰 Technologies Used
- **Python**
- **PyTorch**
- **Transformers (Hugging Face)**
- **Scikit-learn**
- **Pandas**
- **TQDM**

---

## 🤖 Author
**Aneesh M A**  
Fake News Detection Project using BERT and Logistic Regression  
📅 November 2025  

---

## 🪪 License
This project is open source under the [MIT License](LICENSE).
