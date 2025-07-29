import streamlit as st
import fitz  # PyMuPDF
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Streamlit Page Config ---
st.set_page_config(
    page_title="JobVerseAI",
    page_icon="🧠",
    layout="centered"
)

# --- Custom Style ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton > button {
        background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5em 1em;
    }
    .stProgress > div > div > div > div {
        background-color: #00FFAA;
    }
    </style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("<h1 style='text-align:center;'>💼 JobVerseAI</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center;'>AI-Powered Resume & JD Matcher</h4>", unsafe_allow_html=True)
st.write("---")

# --- Upload & Input ---
resume_file = st.file_uploader("📄 Upload Your Resume (PDF)", type=["pdf"])
job_description = st.text_area("📝 Paste the Job Description", height=200)

# --- Function to Extract PDF Text ---
def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# --- Match Score Calculation ---
def get_match_score(resume_text, jd_text):
    vectorizer = TfidfVectorizer().fit_transform([resume_text, jd_text])
    score = cosine_similarity(vectorizer[0:1], vectorizer[1:2])[0][0]
    score = int(score * 100)

    jd_words = set(re.findall(r'\b\w+\b', jd_text.lower()))
    resume_words = set(re.findall(r'\b\w+\b', resume_text.lower()))
    missing = jd_words - resume_words
    keywords = [word for word in missing if len(word) > 4][:10]

    return score, keywords

# --- Button to Check Match ---
if st.button("🔍 Check Resume-JD Match"):
    if resume_file and job_description:
        with st.spinner("Analyzing resume and job description..."):
            resume_text = extract_text_from_pdf(resume_file)
            match_score, missing_skills = get_match_score(resume_text, job_description)

            st.success(f"✅ Match Score: {match_score}%")
            st.progress(match_score)

            if missing_skills:
                st.warning("🔎 Potential missing skills/keywords:")
                for skill in missing_skills:
                    st.markdown(f"- {skill}")
    else:
        st.error("Please upload a resume and paste a job description to proceed.")

# --- Footer ---
st.markdown("---")
st.markdown("<div style='text-align:center;'>Built with ❤️ using Streamlit • 2025 Edition</div>", unsafe_allow_html=True)
