#  Developers Hub Internship — ML Tasks

**Intern:** Saif Ullah  
**Organization:** Developers Hub  

---

## Task 2 — End-to-End ML Pipeline for Customer Churn Prediction

### 🎯 Objective
Build a complete, production-ready machine learning pipeline to predict whether a telecom customer will churn (leave the service), using the IBM Telco Customer Churn dataset.

### 🛠️ Methodology / Approach
- Loaded the Telco Customer Churn dataset (7,043 rows × 21 columns) directly from a public URL.
- Handled missing values in `TotalCharges` by converting to numeric and filling with the median.
- Built a `scikit-learn` `Pipeline` with a `ColumnTransformer` that applies `StandardScaler` to numerical columns and `OneHotEncoder` to categorical columns.
- Trained two models: **Logistic Regression** (baseline) and **Random Forest** (tuned via `GridSearchCV` with 3-fold cross-validation).
- Exported the best model as a `.pkl` file using `joblib` and verified it by reloading and running sample predictions.

### 📊 Key Results / Observations
| Model | Accuracy |
|---|---|
| Logistic Regression | **82.11%** |
| Random Forest (best) | **81.05%** |

- Best Random Forest params: `n_estimators=100`, `max_depth=10`, `min_samples_split=5`
- The model performs well on the majority class (non-churners), with precision of 0.84 and recall of 0.91.
- Churner detection (minority class) is harder — F1-score of 0.59 — highlighting the class imbalance challenge.

---

## Task 3 — Multimodal Housing Price Prediction (Images + Tabular)

### 🎯 Objective
Build a **multimodal deep learning model** that predicts house prices by combining structured tabular features (area, bedrooms, etc.) with image-based visual features — demonstrating how multiple data types can be fused in a single neural network.

### 🛠️ Methodology / Approach
- Generated a synthetic dataset of 1,000 houses with tabular features: `area`, `bedrooms`, `bathrooms`, `location_score`, and `price`.
- Created synthetic 64×64 RGB house images with basic geometric patterns using OpenCV, simulating real property photos.
- Applied `StandardScaler` to tabular features and normalized pixel values to [0, 1].
- Designed a **dual-branch neural network** in Keras:
  - **Branch 1 (Tabular):** Dense layers (128 → 64 units) with Dropout.
  - **Branch 2 (Image / CNN):** Dense layers on 512-dim image feature vectors (256 → 128 units) with Dropout.
  - Both branches are concatenated and passed through final regression layers to output a single price value.
- Also explored a **VGG16-based** variant (Task 4 notebook) for richer image feature extraction.
- Trained with Adam optimizer, MSE loss, for 50 epochs.

### 📊 Key Results / Observations
- The model successfully fuses two different data modalities in a single forward pass.
- MAE and RMSE were computed on the held-out 20% test set.
- Key insight: on synthetic data, tabular features dominate prediction quality; image branches become more impactful with real property photos.
- Demonstrates a scalable architecture pattern applicable to real estate, e-commerce, and medical imaging domains.

---

## Task 4 — Context-Aware Chatbot Using LangChain / RAG

### 🎯 Objective
Design and implement a **context-aware chatbot** that uses Retrieval-Augmented Generation (RAG) principles via LangChain — allowing the bot to answer questions grounded in a custom knowledge base rather than relying solely on a base language model.

### 🛠️ Methodology / Approach
- Leveraged **LangChain** as the orchestration framework to chain together retrieval and generation steps.
- Built a document ingestion pipeline: text documents are split into chunks, converted to vector embeddings, and stored in a vector store.
- On each user query, the most relevant document chunks are retrieved and injected into the prompt context (RAG pattern).
- The language model then generates a response grounded in those retrieved passages — reducing hallucinations and enabling domain-specific Q&A.
- Maintained **conversation memory** across turns so the chatbot is context-aware throughout a session.

### 📊 Key Results / Observations
- The RAG approach significantly improves answer accuracy compared to a plain LLM with no retrieval.
- Context window management (chunking strategy, overlap) proved critical for coherent multi-turn conversations.
- The architecture is modular: the vector store, LLM, and retriever can each be swapped independently (e.g., FAISS → Pinecone, OpenAI → local model).

---

## 🗂️ Repository Structure

```
├── Task_2_End_to_End_ML_Pipeline_for_Customer_Churn_Prediction.ipynb
├── Task_3_Multimodal_Housing_Price_Prediction__Images___Tabular_.ipynb
├── Task_4_Context_Aware_Chatbot_Using_LangChain_RAG.ipynb
└── README.md
```

## 🧰 Tech Stack

`Python` · `scikit-learn` · `TensorFlow / Keras` · `LangChain` · `OpenCV` · `pandas` · `NumPy` · `joblib`
