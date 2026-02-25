import streamlit as st
import lightgbm
import xgboost
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# =============================
# Page config
# =============================
st.set_page_config(
    page_title="Tourism Experience Analytics",
    layout="wide"
)

# =============================
# CACHED LOADERS
# =============================
@st.cache_resource
def load_artifacts():
    reg_pipeline = joblib.load("artifacts/regression_pipeline.pkl")
    clf_pipeline = joblib.load("artifacts/classification_pipeline.pkl")
    label_encoder = joblib.load("artifacts/label_encoder.pkl")
    tfidf = joblib.load("artifacts/tfidf_vectorizer.pkl")
    item_features = joblib.load("artifacts/item_features.pkl")
    rec_df = joblib.load("artifacts/rec_df.pkl")
    item_id_to_pos = joblib.load("artifacts/item_id_to_pos.pkl")

    # rebuild content matrix (lightweight at runtime)
    content_matrix = tfidf.transform(item_features["content"])

    # rebuild user-item matrix
    user_item_matrix = rec_df.pivot_table(
        index="UserId",
        columns="AttractionId",
        values="Rating"
    ).fillna(0)

    user_similarity = cosine_similarity(user_item_matrix)
    user_similarity_df = pd.DataFrame(
        user_similarity,
        index=user_item_matrix.index,
        columns=user_item_matrix.index
    )

    return (
        reg_pipeline,
        clf_pipeline,
        label_encoder,
        tfidf,
        item_features,
        rec_df,
        item_id_to_pos,
        content_matrix,
        user_item_matrix,
        user_similarity_df,
    )

(
    reg_pipeline,
    clf_pipeline,
    label_encoder,
    tfidf,
    item_features,
    rec_df,
    item_id_to_pos,
    content_matrix,
    user_item_matrix_filled,
    user_similarity_df,
) = load_artifacts()

# =============================
# HELPER FUNCTIONS
# =============================

def popularity_fallback(top_n=5):
    popular = (
        rec_df.groupby(["AttractionId", "Attraction"])["Rating"]
        .count()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index(name="VisitCount")
    )
    return popular


def recommend_attractions(user_id, top_n=5):
    if user_id not in user_item_matrix_filled.index:
        return popularity_fallback(top_n)

    similar_users = (
        user_similarity_df[user_id]
        .sort_values(ascending=False)[1:21]
    )

    user_rated = user_item_matrix_filled.loc[user_id]
    unrated_items = user_rated[user_rated == 0].index

    scores = {}

    for item in unrated_items:
        weighted_sum = 0
        sim_sum = 0

        for sim_user, sim_score in similar_users.items():
            rating = user_item_matrix_filled.loc[sim_user, item]
            if rating > 0 and sim_score > 0:
                weighted_sum += sim_score * rating
                sim_sum += sim_score

        if sim_sum > 0:
            scores[item] = weighted_sum / sim_sum

    if len(scores) == 0:
        return popularity_fallback(top_n)

    top_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_ids = [item[0] for item in top_items]

    return (
        item_features[item_features["AttractionId"].isin(top_ids)]
        [["AttractionId", "Attraction"]]
        .drop_duplicates()
    )


def get_similar_items(item_id, top_k=20):
    if item_id not in item_id_to_pos:
        return pd.DataFrame(columns=["AttractionId", "Attraction"])

    idx = item_id_to_pos[item_id]

    sim_scores = cosine_similarity(
        content_matrix[idx], content_matrix
    ).flatten()

    top_indices = sim_scores.argsort()[::-1][1 : top_k + 1]

    return item_features.iloc[top_indices][["AttractionId", "Attraction"]]


def content_recommend(user_id, top_n=10):
    if user_id not in user_item_matrix_filled.index:
        return popularity_fallback(top_n)

    user_ratings = user_item_matrix_filled.loc[user_id]
    liked_items = user_ratings[user_ratings >= 4].index

    if len(liked_items) == 0:
        return popularity_fallback(top_n)

    recs = []
    for item in liked_items[:3]:
        similar = get_similar_items(item, top_k=top_n)
        recs.append(similar)

    content_recs = pd.concat(recs).drop_duplicates().head(top_n)
    return content_recs


def hybrid_recommend(user_id, top_n=5, alpha=0.6, beta=0.4):
    cf_recs = recommend_attractions(user_id, top_n=top_n * 3)
    if "AttractionId" not in cf_recs.columns:
        cf_recs = popularity_fallback(top_n * 3)

    cf_recs = cf_recs[["AttractionId", "Attraction"]].copy()
    cf_recs["score"] = alpha

    content_recs = content_recommend(user_id, top_n=top_n * 3)
    content_recs = content_recs[["AttractionId", "Attraction"]].copy()
    content_recs["score"] = beta

    combined = pd.concat([cf_recs, content_recs], ignore_index=True)

    final = (
        combined.groupby(["AttractionId", "Attraction"])["score"]
        .sum()
        .reset_index()
        .sort_values("score", ascending=False)
        .head(top_n)
    )

    if len(final) == 0:
        return popularity_fallback(top_n)

    return final[["AttractionId", "Attraction"]]

# =============================
# UI
# =============================
st.title("🌍 Tourism Experience Analytics Dashboard")

menu = st.sidebar.radio(
    "Navigation",
    ["Visit Mode Prediction", "Rating Prediction", "Hybrid Recommender"],
)

# =============================
# CLASSIFICATION
# =============================
if menu == "Visit Mode Prediction":
    st.header("🧭 Visit Mode Prediction")

    input_df = st.text_area(
        "Paste feature JSON (same format as training features)",
        "{}",
        height=150,
    )

    if st.button("Predict Visit Mode"):
        try:
            data = pd.read_json(input_df)
            pred = clf_pipeline.predict(data)
            label = label_encoder.inverse_transform(pred)
            st.success(f"Predicted Visit Mode: {label[0]}")
        except Exception as e:
            st.error(f"Error: {e}")

# =============================
# REGRESSION
# =============================
elif menu == "Rating Prediction":
    st.header("⭐ Rating Prediction")

    input_df = st.text_area(
        "Paste feature JSON (same format as training features)",
        "{}",
        height=150,
    )

    if st.button("Predict Rating"):
        try:
            data = pd.read_json(input_df)
            pred = reg_pipeline.predict(data)
            st.success(f"Predicted Rating: {pred[0]:.2f}")
        except Exception as e:
            st.error(f"Error: {e}")

# =============================
# RECOMMENDER
# =============================
else:
    st.header("🎯 Hybrid Attraction Recommender")

    user_id = st.number_input("Enter User ID", min_value=0, step=1)
    top_n = st.slider("Number of recommendations", 3, 10, 5)

    if st.button("Get Recommendations"):
        recs = hybrid_recommend(user_id, top_n=top_n)
        st.dataframe(recs.reset_index(drop=True))

