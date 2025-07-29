import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pdfplumber
import tempfile

st.set_page_config(page_title="JobVerse AI - Resume & Job Matcher", layout="wide")

model = SentenceTransformer('all-MiniLM-L6-v2')

st.title("🌌 JobVerse AI")
st.markdown("Match your resume with any job description using advanced AI and NLP.")

uploaded_file = st.file_uploader("📄 Upload your Resume (PDF format)", type="pdf")
job_description = st.text_area("💼 Paste the Job Description", height=200)

def extract_text_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        return "\n".join([page.extract_text() or "" for page in pdf.pages])

if uploaded_file and job_description:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    resume_text = extract_text_from_pdf(tmp_path)

    embeddings = model.encode([resume_text, job_description])
    similarity_score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0] * 100

    st.subheader("📊 Match Score")
    st.metric(label="Resume-JD Fit", value=f"{similarity_score:.2f}%")

    if similarity_score > 75:
        st.success("✅ Excellent fit! Your resume closely matches the job description.")
    elif similarity_score > 50:
        st.warning("⚠️ Moderate fit. Consider tailoring your resume further.")
    else:
        st.error("❌ Low match. You might want to enhance your resume content.")

    with st.expander("📄 Extracted Resume Text"):
        st.write(resume_text)

    with st.expander("💼 Job Description Provided"):
        st.write(job_description)
