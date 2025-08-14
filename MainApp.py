import streamlit as st
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# --- Load Data ---
df = pd.read_csv("DataSet/Symptom2DiseaseFix.csv")

# Pastikan kolom sesuai
text_col = 'text'
label_col = 'disease'

X = df[text_col]
y = df[label_col]

# --- Vectorization ---
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# --- Train-test split ---
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# --- Models ---
models = {
    "SVM": LinearSVC(),
    "Naive Bayes": MultinomialNB(),
    "Random Forest": RandomForestClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier()
}

# Train semua model
for name, model in models.items():
    model.fit(X_train, y_train)

# --- UI ---
st.set_page_config(page_title="Prediksi Penyakit", layout="wide")
with open("front/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("Prediksi Penyakit Berdasarkan Deskripsi")

tab1, tab2, tab3 = st.tabs(["Prediksi", "Hasil Per Model", "Persentase Kemungkinan"])

with tab1:
    deskripsi = st.text_area("Deskripsi Gejala")
    model_choice = st.selectbox("Pilih Model", list(models.keys()))

    if st.button("Prediksi"):
        if deskripsi.strip():
            # Create placeholder for loader
            loader_placeholder = st.empty()

            # Show loading animation
            loader_placeholder.markdown(
                """
                <div style="text-align:center;">
                    <img src="https://media.giphy.com/media/L05HgB2h6qICDs5Sms/giphy.gif" 
                         width="150" alt="Loading...">
                    <p><b>🔍 Menganalisis gejala Anda...</b></p>
                </div>
                """,
                unsafe_allow_html=True
            )

            time.sleep(2)  # Simulasi loading

            # Prediction
            deskripsi_vec = vectorizer.transform([deskripsi])
            pred = models[model_choice].predict(deskripsi_vec)[0]

            # Remove loader
            loader_placeholder.empty()

            # Show result
            st.success(f"🩺 Hasil Prediksi ({model_choice}): {pred}")
        else:
            st.warning("Masukkan deskripsi gejala terlebih dahulu.")

with tab2:
    st.subheader("Hasil Prediksi dari Semua Model")
    if deskripsi.strip():
        deskripsi_vec = vectorizer.transform([deskripsi])
        results = {name: mdl.predict(deskripsi_vec)[0] for name, mdl in models.items()}
        st.table(pd.DataFrame(results.items(), columns=["Model", "Prediksi"]))
    else:
        st.info("Masukkan deskripsi di tab Prediksi.")

with tab3:
    st.subheader("Persentase Kemungkinan")
    if deskripsi.strip():
        deskripsi_vec = vectorizer.transform([deskripsi])
        percents = {}
        for name, mdl in models.items():
            try:
                prob = mdl.predict_proba(deskripsi_vec)[0]
                prob_dict = dict(zip(mdl.classes_, prob))
                pred, max_prob = max(prob_dict.items(), key=lambda x: x[1])
                percents[name] = f"{pred} ({max_prob:.2%})"
            except:
                percents[name] = "Tidak tersedia"
        st.table(pd.DataFrame(list(percents.items()), columns=["Model", "Prediksi & Probabilitas"]))
    else:
        st.info("Masukkan deskripsi di tab Prediksi.")
