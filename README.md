# 🟣 News Topic Discovery Dashboard 🌳  
**Hierarchical Clustering for Automatic Topic Discovery**

---

## 🔹 Overview

The **News Topic Discovery Dashboard** is an interactive Streamlit application that automatically groups similar news articles using **Hierarchical (Agglomerative) Clustering**.

Instead of manually defining categories, this system discovers hidden themes directly from textual patterns. It helps editors and analysts understand emerging topics without labeled data.

This project demonstrates practical **unsupervised machine learning applied to real-world editorial intelligence**.

---

## 🎯 Objective

- Automatically group similar news articles  
- Discover hidden themes without predefined labels  
- Visualize topic structure using dendrograms  
- Provide business-friendly cluster interpretation  

---

## 📚 Concepts Implemented

- 🧠 Unsupervised Learning  
- 📝 Text Preprocessing  
- 🔠 TF-IDF Vectorization  
- 🌳 Agglomerative (Hierarchical) Clustering  
- 📈 Dendrogram Analysis  
- 📉 PCA for Dimensionality Reduction  
- 📊 Silhouette Score for Cluster Validation  
- 🏢 Business Interpretation of Clusters  

---

## 🛠 Tech Stack

- Python  
- Streamlit  
- Scikit-learn  
- SciPy  
- Pandas  
- NumPy  
- Matplotlib  
- Plotly  

---

## 🌳 Key Features

### 1️⃣ Text Vectorization Controls
- Adjustable TF-IDF feature limit  
- Stopword removal option  
- N-gram selection (Unigrams / Bigrams / Both)

### 2️⃣ Hierarchical Clustering Controls
- Multiple linkage methods:
  - Ward  
  - Complete  
  - Average  
  - Single  
- Subset selection for dendrogram visualization  

### 3️⃣ Dendrogram Visualization
- Displays hierarchical cluster tree  
- Helps identify natural cluster separations  
- Optional cut-height inspection  

### 4️⃣ Cluster Application
- User-defined number of clusters  
- Real-time clustering updates  

### 5️⃣ PCA-Based 2D Visualization
- Projects high-dimensional text into 2D  
- Interactive scatter plot  
- Color-coded clusters  
- Hover preview of article snippets  

### 6️⃣ Cluster Summary Table
For each cluster:
- Cluster ID  
- Number of articles  
- Top keywords  
- Representative article snippet  

### 7️⃣ Validation Metric
**Silhouette Score** is displayed to measure clustering quality.

Score interpretation:
- Close to 1 → Well-separated clusters  
- Around 0 → Overlapping clusters  
- Negative → Poor clustering  

### 8️⃣ Business Insight Section
Clusters are explained in non-technical language to highlight:
- Editorial themes  
- Content categorization opportunities  
- Recommendation system potential  

---

## 🏢 Business Applications

- 🏷 Automatic news tagging  
- 🔎 Topic discovery  
- 📚 Content organization  
- 🤖 Recommendation systems  
- 📰 Editorial workflow optimization  

Articles grouped together share similar vocabulary and thematic structure.

---

## 🚀 How to Run

1. Install dependencies:
