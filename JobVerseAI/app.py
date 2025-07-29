import streamlit as st
import pdfplumber
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import tempfile

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Set Page config
st.set_page_config(page_title="JobVerseAI", layout="centered", page_icon="🧠")

st.markdown("<h1 style='text-align: center;'>🧠 JobVerseAI</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>AI-Powered Resume & Job Description Matcher</h4>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.title("📂 Upload Your Files")
    resume_file = st.file_uploader("📄 Upload Resume (PDF)", type="pdf")
    st.markdown("---")
    jd_text = st.text_area("💼 Paste Job Description")

# Main Section
if st.button("🔍 Check Match", use_container_width=True):
    if resume_file is None or jd_text.strip() == "":
        st.error("Please upload a resume and enter a job description.")
    else:
        with st.spinner("Analyzing match..."):
            # Extract resume text
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp:
                temp.write(resume_file.read())
                temp_path = temp.name

            with pdfplumber.open(temp_path) as pdf:
                resume_text = ""
                for page in pdf.pages:
                    resume_text += page.extract_text()

            # Get embeddings
            resume_embedding = model.encode([resume_text])[0]
            jd_embedding = model.encode([jd_text])[0]

            # Cosine similarity
            score = cosine_similarity([resume_embedding], [jd_embedding])[0][0]
            percent_match = round(score * 100, 2)

        st.success("✅ Match analysis complete!")

        # Show match
        st.markdown(f"""
        <h2 style='text-align: center; color: green;'>🔗 Match Score: {percent_match}%</h2>
        """, unsafe_allow_html=True)

        st.progress(int(percent_match))

        # Feedback section
        if percent_match > 80:
            st.success("🎉 Great match! Your resume aligns well with the job.")
        elif percent_match > 60:
            st.warning("🧐 Decent match. Consider improving your skills section or work experience.")
        else:
            st.error("❗Low match. Tailor your resume to better reflect the job description.")

        # Optional resume tips
        with st.expander("📌 Suggestions to Improve Resume"):
            st.markdown("""
            - Match job keywords exactly (e.g., "data pipelines", "Kubernetes")
            - Highlight relevant experiences and results (quantified)
            - Include certifications or side-projects aligned with the role
            - Use action words: *led*, *built*, *automated*, *deployed*
            """)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center;'>Built with ❤️ by <a href='https://github.com/BhavethraBaabu'>Bhavethra Baabu</a></p>", unsafe_allow_html=True)
