import streamlit as st
import pdfplumber
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="JobVerseAI", layout="centered")

st.markdown("""
    <style>
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        padding: 8px 20px;
        border-radius: 10px;
    }
    .stProgress > div > div > div > div {
        background-color: #00FFAA;
    }
    </style>
""", unsafe_allow_html=True)

st.title("💼 JobVerseAI")
st.markdown("**AI-Powered Resume & Job Description Matcher**")
st.write("---")

resume_file = st.file_uploader("📄 Upload Your Resume (PDF)", type=["pdf"])
job_description = st.text_area("📝 Paste the Job Description", height=200)

def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def get_match_score(resume_text, jd_text):
    vectorizer = TfidfVectorizer().fit_transform([resume_text, jd_text])
    score = cosine_similarity(vectorizer[0:1], vectorizer[1:2])[0][0]
    score = int(score * 100)

    jd_words = set(re.findall(r'\b\w+\b', jd_text.lower()))
    resume_words = set(re.findall(r'\b\w+\b', resume_text.lower()))
    missing = jd_words - resume_words
    keywords = [word for word in missing if len(word) > 4][:10]

    return score, keywords

if st.button("🔍 Check Resume-JD Match"):
    if resume_file and job_description:
        with st.spinner("Analyzing..."):
            resume_text = extract_text_from_pdf(resume_file)
            match_score, missing_skills = get_match_score(resume_text, job_description)

            st.success(f"✅ Match Score: {match_score}%")
            st.progress(match_score)

            if missing_skills:
                st.warning("🚫 Missing Keywords:")
                st.markdown(", ".join(missing_skills))
    else:
        st.error("Please upload a resume and paste a JD.")

st.markdown("---")

