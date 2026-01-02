import streamlit as st
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import docx

# Load English NLP model
nlp = spacy.load("en_core_web_sm")

# --- 1. Predefined Technical Skill Dictionary ---
# You can expand this list with more skills relevant to your domain.
SKILL_DB = {
    # Programming Languages
    "python", "java", "c++", "c", "c#", "javascript", "typescript", "ruby", "php", "swift", "go", "rust", "sql", "html", "css",
    # Frameworks & Libraries
    "react", "angular", "vue", "node.js", "django", "flask", "fastapi", "spring", "dotnet", "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "keras", "opencv", "nltk", "spacy",
    # Cloud & DevOps
    "aws", "azure", "google cloud", "gcp", "docker", "kubernetes", "jenkins", "terraform", "ansible", "git", "github", "gitlab", "ci/cd", "linux", "unix",
    # Data & Databases
    "mysql", "postgresql", "mongodb", "redis", "oracle", "tableau", "power bi", "excel", "spark", "hadoop", "kafka",
    # Concepts & Others
    "machine learning", "deep learning", "nlp", "computer vision", "data analysis", "agile", "scrum", "rest api", "graphql", "microservices", "oop", "system design"
}

def extract_text_from_pdf(file):
    """Extracts text from an uploaded PDF file."""
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_docx(file):
    """Extracts text from an uploaded DOCX file."""
    doc = docx.Document(file)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

def preprocess_text(text):
    """
    Cleans text: removes stopwords, punctuation, and performs lemmatization.
    Returns the cleaned text string.
    """
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and not token.is_space]
    return " ".join(tokens)

def extract_keywords(text):
    """
    Extracts keywords by matching text against the predefined SKILL_DB.
    Handles both single-word tokens and multi-word phrases (noun chunks).
    """
    doc = nlp(text.lower())
    found_skills = set()

    # 1. Check individual tokens (e.g., "Python", "Java")
    for token in doc:
        if token.text in SKILL_DB or token.lemma_ in SKILL_DB:
            found_skills.add(token.text)

    # 2. Check noun chunks for multi-word skills (e.g., "Machine Learning", "Data Analysis")
    for chunk in doc.noun_chunks:
        if chunk.text in SKILL_DB or chunk.lemma_ in SKILL_DB:
            found_skills.add(chunk.text)
            
    return found_skills

# --- Streamlit UI Layout ---
st.set_page_config(page_title="Resume Matcher", page_icon="📄")

st.title("🤖 AI-Powered Resume Matcher")
st.markdown("Upload your **Resume** and the **Job Description** to get a compatibility score and find missing keywords.")

st.markdown("---")

# Columns for file uploaders
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Upload Resume")
    resume_file = st.file_uploader("Upload PDF or DOCX", type=["pdf", "docx"])

with col2:
    st.subheader("2. Upload Job Description")
    jd_file = st.file_uploader("Upload PDF, DOCX, or Paste Text", type=["pdf", "docx"])
    jd_text_input = st.text_area("Or paste JD text here:", height=150)

st.markdown("---")

if st.button("Analyze Match"):
    if resume_file and (jd_file or jd_text_input):
        with st.spinner("Processing documents..."):
            
            # 1. Extract Resume Text
            if resume_file.name.endswith(".pdf"):
                resume_text = extract_text_from_pdf(resume_file)
            else:
                resume_text = extract_text_from_docx(resume_file)

            # 2. Extract JD Text
            if jd_file:
                if jd_file.name.endswith(".pdf"):
                    jd_text_raw = extract_text_from_pdf(jd_file)
                else:
                    jd_text_raw = extract_text_from_docx(jd_file)
            else:
                jd_text_raw = jd_text_input

            # 3. Preprocess Texts
            clean_resume = preprocess_text(resume_text)
            clean_jd = preprocess_text(jd_text_raw)

            # 4. Calculate Similarity using TF-IDF and Cosine Similarity
            data = [clean_resume, clean_jd]
            tfidf_vectorizer = TfidfVectorizer()
            tfidf_matrix = tfidf_vectorizer.fit_transform(data)
            
            # Similarity between Resume (index 0) and JD (index 1)
            similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
            match_score = similarity_matrix[0][0] * 100

            # 5. Keyword Analysis (Using SKILL_DB)
            resume_keywords = extract_keywords(resume_text)
            jd_keywords = extract_keywords(jd_text_raw)
            missing_keywords = jd_keywords - resume_keywords

            # --- Display Results ---
            st.success("Analysis Complete!")
            
            # Score Display
            st.subheader(f"Match Score: {match_score:.2f}%")
            st.progress(int(match_score))
            
            # Contextual Feedback based on score
            if match_score >= 80:
                st.balloons()
                st.info("Great match! Your resume covers most of the requirements.")
            elif match_score >= 50:
                st.warning("Good potential, but you are missing some key requirements.")
            else:
                st.error("Low match. Consider rewriting your resume to target this role.")

            # Missing Keywords Display
            st.subheader("⚠️ Missing Skills found in JD:")
            if missing_keywords:
                st.write("Consider adding these technical skills to your resume:")
                # Display as chips/tags
                st.write(", ".join([f"`{kw}`" for kw in list(missing_keywords)]))
            else:
                st.write("No major skills missing! Great job.")

    else:
        st.error("Please upload both the Resume and the Job Description to proceed.")