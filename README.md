# 💊 Drug-Target Affinity Prediction Using Deep Learning

## 🚀 Project Overview  
This project implements a deep learning pipeline to predict drug-target binding affinities, inspired by the **DeepDTA** architecture.  
- **Drugs** are represented by *encoded canonical* and *isomeric SMILES*.  
- **Proteins** are represented by *numeric sequence encodings*.  
The goal is to accurately regress molecular affinities for use in drug discovery applications.

---

## 📦 Features  
- 🔬 **Advanced Encodings**: Canonical & isomeric SMILES + protein sequence encoding  
- 🧹 **Robust Preprocessing**: Parses stringified vectors, handles missing values, and standardizes feature shapes  
- 🧠 **Deep Multi-Input Model**: Three-branch neural network for different feature types (SMILES and sequences)  
- 📈 **Benchmark Evaluation**: Reports **Mean Squared Error (MSE)** and **Mean Absolute Error (MAE)**  

---

## 📂 Dataset  
`Finalized_dataset.csv` contains:  
- 🧪 Drug Features: `canonical_*` and `isomeric_*` columns (stringified lists)  
- 🧬 Protein Features: `sequence_*` columns (stringified lists)  
- 🎯 Target Affinity: `Affinity` column (range: **5.0 to 10.8**)  

> ⚠️ Missing values are filled with zeros during preprocessing.

---

## ⚙️ Setup & Installation

### 1. Clone the Repository
```
bash
git clone https://github.com/yourusername/drug-target-affinity-prediction.git
cd drug-target-affinity-prediction
```

### 2. Create and Activate Virtual Environment
```
bash
# Create virtual environment
python -m venv venv

# Activate it:
# On Windows
venv\Scripts\activate.bat

# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```
bash
pip install -r requirements.txt
```

🏃‍♂️ Usage
1. Data Preparation
Ensure Finalized_dataset.csv is present in the project root.
Encoded features must be in string-list format. They’ll be parsed automatically.

2. Train the Model
```
bash
python deepDTA.py
```
The script will:
Parse and convert all feature columns to numeric arrays
Build and train the deep learning model
Output evaluation metrics

3. Evaluation Example
```
text
Test Loss (MSE): 0.49  
Test MAE: 0.42  
```
This reflects ~4–8% error, based on the affinity range (5.0 – 10.8).

🗂️ Project Structure
```
bash
├── Finalized_dataset.csv    # Main dataset
├── deepDTA.py               # Main pipeline: preprocessing, training, evaluation
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation (this file)
├── utils.py                 # (Optional) Helper scripts for parsing
└── models.py                # (Optional) Neural network model definitions
```

📊 Results
| Metric | Value |
|--------|-------|
|Test MAE| ~0.42 |
|Test MSE| ~0.49 |

Based on affinity range 5.0–10.8, this indicates strong prediction accuracy.

🔬 Notes & Recommendations
✅ Data parser handles missing or inconsistent shapes robustly.

🛠️ Easily extensible to:

Pretrained embeddings
Alternative model architectures
Large-scale dataset scaling (e.g., generators, sharding)

🙏 Acknowledgements

📚 Raw Dataset by @dingyan20 (https://github.com/dingyan20/Davis-Dataset-for-DTA-Prediction?tab=readme-ov-file)

🔧 smiles_encoder library by @tjkessler (https://github.com/tjkessler/smiles-encoder)
