<p align="center">
  <img src="https://raw.githubusercontent.com/Naziyasompalem/fetal-health-GCN/main/assets/Thumbnail.jpg" 
       alt="Fetal Health Prediction Thumbnail" width="650">
</p>

# Fetal Health Prediction using Graph Convolutional Networks (GCNs)

This project predicts fetal health status using Graph Neural Networks
(GCNs) based on CTG (Cardiotocography) data.\
The goal is to classify fetal health into three categories: **Normal**,
**Suspect**, and **Pathological**.

------------------------------------------------------------------------

## 🚀 Features

-   Preprocessing of CTG dataset (handling missing values,
    normalization)
-   Graph construction for GCN-based learning
-   GCN model implementation using PyTorch Geometric
-   Training + Evaluation pipeline
-   Model metrics visualization (accuracy, confusion matrix)
-   Easy-to-run notebook-based workflow

------------------------------------------------------------------------

## 📁 Project Structure

    Fetal-Health-GCN/
    │── data/
    │   ├── CTGData
    │   ├──CTGData_cleaned
    │   ├──X_test
    │   ├──X_train
    │   ├──y_test
    │   └──y_train
    │── src/
    │   ├── preprocess.py
    │   ├── utils.py
    │   ├── model.py
    │   └── train.py
    │── notebooks/
    │   ├── GCN3layer
    │   └── Feature_selection
    │── README.md
    │── requirements.txt
    │── .gitignore

------------------------------------------------------------------------

## 🧠 Model Workflow

1.  **Load CTG dataset**\
2.  **Clean + Normalize features**\
3.  **Build graph using correlation between features**\
4.  **Train GCN model**\
5.  **Evaluate accuracy + loss + confusion matrix**

------------------------------------------------------------------------

## 📊 Dataset

We use the **Fetal Health Classification** dataset, available on Kaggle
& UCI ML Repository.

Features include: - Baseline value\
- Accelerations\
- Fetal movement\
- Uterine contractions\
- Abnormal short term variability\
- ... and more

Target class:\
- `1` Normal\
- `2` Suspect\
- `3` Pathological

------------------------------------------------------------------------

## 🛠️ Technologies Used

-   Python
-   PyTorch
-   PyTorch Geometric
-   NumPy / Pandas
-   Scikit-Learn
-   Matplotlib / Seaborn

------------------------------------------------------------------------

## ▶️ How to Run

### 1️⃣ Clone the repository

    git clone https://github.com/Naziyasompalem/fetal-health-GCN.git
    cd Fetal-Health-GCN

### 2️⃣ Install dependencies

    pip install -r requirements.txt

### 3️⃣ Run Jupyter notebook

    jupyter notebook

### 4️⃣ Train the model

Open:\
`notebooks/Fetal_Health_GCN.ipynb`

------------------------------------------------------------------------

## 📈 Results

-   Achieved high accuracy using GCN-based approach\
-   GNN performed better than traditional ML models\
-   Visualized metrics help interpret performance

------------------------------------------------------------------------

## 🤝 Contributing

Contributions are welcome!\
Create a pull request or open an issue if you want to improve something.

------------------------------------------------------------------------

## 📜 License

This project is licensed under the **MIT License**.

------------------------------------------------------------------------

## 💬 Contact

If you have any questions, feel free to reach out!\
**Author:** Naziya\
**Email:** naziyasompalem@gmail.com
