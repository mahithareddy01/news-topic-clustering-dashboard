import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram

st.set_page_config(layout="wide")
st.title("🟣 News Topic Discovery Dashboard")
st.markdown(
    "This system uses **Hierarchical Clustering** to automatically group similar news articles based on textual similarity."
)

# ================================
# LOAD DATASET (HARDCODED)
# ================================

@st.cache_data
def load_data():
    df = pd.read_csv("all-data.csv", encoding="latin1")  
    return df

df = load_data()

# Auto-detect text column
text_cols = df.select_dtypes(include='object').columns

# Automatically pick longest text column
avg_lengths = {col: df[col].astype(str).str.len().mean() for col in text_cols}
text_column = max(avg_lengths, key=avg_lengths.get)

st.sidebar.write(f"Detected Text Column: **{text_column}**")

df = df.dropna(subset=[text_column])
texts = df[text_column].astype(str)

# ================================
# SIDEBAR CONTROLS
# ================================

st.sidebar.header("📝 TF-IDF Settings")

max_features = st.sidebar.slider("Max TF-IDF Features", 100, 2000, 1000)
remove_stopwords = st.sidebar.checkbox("Use English Stopwords", value=True)

ngram_option = st.sidebar.selectbox(
    "N-gram Range",
    ["Unigrams", "Bigrams", "Unigrams + Bigrams"]
)

ngram_map = {
    "Unigrams": (1, 1),
    "Bigrams": (2, 2),
    "Unigrams + Bigrams": (1, 2)
}

vectorizer = TfidfVectorizer(
    max_features=max_features,
    stop_words="english" if remove_stopwords else None,
    ngram_range=ngram_map[ngram_option]
)

X = vectorizer.fit_transform(texts)

# ================================
# HIERARCHICAL SETTINGS
# ================================

st.sidebar.header("🌳 Hierarchical Clustering")

linkage_method = st.sidebar.selectbox(
    "Linkage Method",
    ["ward", "complete", "average", "single"]
)

sample_size = st.sidebar.slider(
    "Articles for Dendrogram",
    20,
    min(200, len(df)),
    min(100, len(df))
)

# ================================
# DENDROGRAM SECTION
# ================================

if st.button("🟦 Generate Dendrogram"):

    st.subheader("🌳 Dendrogram")

    X_dense = X.toarray()
    sample_indices = np.random.choice(len(X_dense), sample_size, replace=False)
    X_sample = X_dense[sample_indices]

    Z = linkage(X_sample, method=linkage_method)

    fig, ax = plt.subplots(figsize=(12, 6))
    dendrogram(Z, ax=ax)
    ax.set_ylabel("Distance")
    ax.set_xlabel("Article Index")

    st.pyplot(fig)

    max_height = float(max(Z[:, 2]))
    cut_height = st.slider("Cut Height", 0.0, max_height, max_height / 2)

    fig2, ax2 = plt.subplots(figsize=(12, 6))
    dendrogram(Z, ax=ax2)
    ax2.axhline(y=cut_height, color='red', linestyle='--')
    ax2.set_ylabel("Distance")
    ax2.set_xlabel("Article Index")

    st.pyplot(fig2)

# ================================
# APPLY CLUSTERING
# ================================

st.subheader("🟩 Apply Clustering")

n_clusters = st.slider("Number of Clusters", 2, 15, 4)

model = AgglomerativeClustering(
    n_clusters=n_clusters,
    linkage=linkage_method
)

labels = model.fit_predict(X.toarray())
df["Cluster"] = labels

# ================================
# PCA VISUALIZATION
# ================================

st.subheader("📉 2D Cluster Projection (PCA)")

pca = PCA(n_components=2)
X_2d = pca.fit_transform(X.toarray())

plot_df = pd.DataFrame({
    "PC1": X_2d[:, 0],
    "PC2": X_2d[:, 1],
    "Cluster": labels,
    "Text": texts.str[:150]
})

fig = px.scatter(
    plot_df,
    x="PC1",
    y="PC2",
    color="Cluster",
    hover_data=["Text"],
    title="News Clusters (2D Projection)"
)

st.plotly_chart(fig, use_container_width=True)

# ================================
# CLUSTER SUMMARY
# ================================

st.subheader("📊 Cluster Summary")

terms = vectorizer.get_feature_names_out()
summary_data = []

X_dense = X.toarray()

for i in range(n_clusters):
    cluster_texts = X_dense[labels == i]
    count = len(cluster_texts)

    if count > 0:
        mean_tfidf = cluster_texts.mean(axis=0)
        top_indices = mean_tfidf.argsort()[-10:][::-1]
        top_terms = [terms[j] for j in top_indices]

        representative = df[df["Cluster"] == i][text_column].iloc[0][:200]

        summary_data.append({
            "Cluster ID": i,
            "Number of Articles": count,
            "Top Keywords": ", ".join(top_terms),
            "Sample Article": representative
        })

summary_df = pd.DataFrame(summary_data)
st.dataframe(summary_df)

# ================================
# SILHOUETTE SCORE
# ================================

st.subheader("📈 Validation")

score = silhouette_score(X_dense, labels)
st.metric("Silhouette Score", round(score, 3))

st.markdown("""
**Interpretation**
- Close to 1 → Well-separated clusters  
- Close to 0 → Overlapping clusters  
- Negative → Poor clustering  
""")

# ================================
# BUSINESS INTERPRETATION
# ================================

st.subheader("🏢 Business Insights")

for row in summary_data:
    main_keyword = row["Top Keywords"].split(",")[0]
    st.markdown(
        f"🟣 **Cluster {row['Cluster ID']}** focuses mainly on articles related to **{main_keyword}** and similar themes."
    )

# ================================
# INSIGHT BOX
# ================================

st.info(
    "Articles grouped in the same cluster share similar vocabulary and themes. "
    "These clusters can be used for automatic tagging, recommendations, and content organization."
)