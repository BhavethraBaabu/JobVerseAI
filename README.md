# JobVerseAI - AI-Powered Resume & Job Matcher
JobVerseAI is your intelligent career companion. It uses state-of-the-art NLP (Sentence Transformers) to match your resume against any job description, providing a match percentage and deep insights — all powered by open-source AI.






---

(https://job-verseai.streamlit.app/)  


---

🧠 How It Works

- 🔍 Extracts text from your uploaded **PDF resume**
- 💬 Accepts any **job description**
- 🤖 Uses **`all-MiniLM-L6-v2`** from Hugging Face for semantic embeddings
- 📊 Computes **cosine similarity** between resume and job description

---

 🛠 Tech Stack

- **Python**
- **Streamlit**
- **HuggingFace Sentence Transformers**
- **Scikit-learn**
- **PDFplumber**

---

💡 Use Cases

- ✅ Job seekers comparing their resume to listings
- 🧑‍💼 Recruiters filtering best-fit resumes
- 🎓 Students applying to internships

---

🧰 Installation

```bash
git clone https://github.com/BhavethraBaabu/JobVerseAI.git
cd JobVerseAI
pip install -r requirements.txt
streamlit run app.py
