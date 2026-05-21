import streamlit as st
from PyPDF2 import PdfReader
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="centered"
)

skills_db = [
    "python",
    "machine learning",
    "deep learning",
    "sql",
    "tensorflow",
    "pytorch",
    "react",
    "java",
    "aws",
    "docker",
    "kubernetes",
    "nlp",
    "data analysis",
    "flask",
    "fastapi",
    "streamlit",
    "git",
    "github",
    "linux",
    "javascript",
    "html",
    "css"
]

def clean_text(text):

    text = re.sub(r'[^a-zA-Z0-9\n ]', ' ', text)

    text = re.sub(r'\s+', ' ', text)

    text = text.lower()

    return text.strip()

def format_resume_text(text):

    text = re.sub(r'[^a-zA-Z0-9.,\n ]', ' ', text)

    text = re.sub(r' +', ' ', text)

    text = re.sub(r'\n+', '\n', text)

    return text.strip()

st.title("📄 AI Resume Analyzer")

st.markdown(
    "Analyze your resume using NLP and Machine Learning"
)

uploaded_file = st.file_uploader(
    "Upload Your Resume",
    type=["pdf"]
)

job_description = st.text_area(
    "Paste Job Description"
)

if uploaded_file is not None and job_description != "":

    reader = PdfReader(uploaded_file)

    resume_text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            resume_text += page_text + "\n"

    formatted_resume = format_resume_text(resume_text)

    cleaned_resume = clean_text(resume_text)

    cleaned_jd = clean_text(job_description)

    texts = [cleaned_resume, cleaned_jd]

    vectorizer = TfidfVectorizer()

    vectors = vectorizer.fit_transform(texts)

    similarity = cosine_similarity(vectors[0], vectors[1])

    score = similarity[0][0] * 100

    st.subheader("📊 Resume Match Score")

    st.metric(
        label="Matching Percentage",
        value=f"{score:.2f}%"
    )

    st.progress(int(score))

    if score >= 80:
        feedback = "Excellent match for the job!"

    elif score >= 60:
        feedback = "Good match, but resume can be improved."

    else:
        feedback = "Low match. Add more relevant skills and experience."

    st.subheader("🧠 ATS Feedback")

    st.info(feedback)

    resume_skills = []

    for skill in skills_db:

        if skill in cleaned_resume:
            resume_skills.append(skill)

    st.subheader("✅ Detected Skills")

    if resume_skills:

        for skill in resume_skills:
            st.success(skill)

    else:
        st.warning("No skills detected.")

    jd_skills = []

    for skill in skills_db:

        if skill in cleaned_jd:
            jd_skills.append(skill)

    missing_skills = []

    for skill in jd_skills:

        if skill not in resume_skills:
            missing_skills.append(skill)

    st.subheader("❌ Missing Skills")

    if missing_skills:

        for skill in missing_skills:
            st.error(skill)

    else:
        st.success("No missing skills detected!")

    with st.expander("📄 View Cleaned Resume Text"):

        st.text_area(
            "Formatted Resume",
            formatted_resume,
            height=400
        )

else:

    st.warning(
        "Please upload a resume and paste a job description."
    )