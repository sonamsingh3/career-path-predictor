import streamlit as st
import pickle
import PyPDF2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
# ---------------- Page Config ----------------
st.set_page_config(
    page_title="AI Career Predictor",
    page_icon="logo.png",
    layout="wide"
)

# ---------------- Header Section ----------------
col1, col2, col3 = st.columns([1,2,1])

with col2:
    st.image("logo.png", width=180)
    st.markdown(
        "<h1 style='text-align: center; color: #4CAF50;'>AI Smart Career Path Predictor</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align: center;'>Upload your resume and get AI-based career suggestions instantly</p>",
        unsafe_allow_html=True
    )

st.markdown("---")

# ---------------- Sidebar ----------------
st.sidebar.image("logo.png", width=120)
st.sidebar.title("About")
st.sidebar.info("This AI model predicts the best career path based on your resume skills.")

# ---------------- Footer ----------------
st.markdown("---")
st.markdown("<center>Made with ❤️ by Sonam Singh</center>", unsafe_allow_html=True)

st.markdown("""
    <style>
    .stApp {
        background-color: #f5f7fa;
    }
    .block-container {
        padding-top: 2rem;
    }
    h1, h2, h3 {
        color: #1f4e79;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- CAREER INFO ----------------
career_info = {
    "Data Scientist": {
        "skills": ["Python","Machine Learning","Data Analysis","SQL"],
        "salary": "₹8–20 LPA",
        "description": "Works with data to build predictive models and insights."
    },
    "Software Engineer": {
        "skills": ["Python","Java","SQL"],
        "salary": "₹5–15 LPA",
        "description": "Designs and develops scalable software systems."
    },
    "Software Developer": {
        "skills": ["Python","SQL","Problem Solving"],
        "salary": "₹5–15 LPA",
        "description": "Builds and maintains applications."
    },
    "Business Analyst": {
        "skills": ["Communication","Excel","Power BI"],
        "salary": "₹6–18 LPA",
        "description": "Analyzes business data and improves processes."
    }
}

skills_list = [
    "python","machine learning","sql","communication",
    "leadership","data analysis","deep learning",
    "excel","power bi","java","problem solving"
]

soft_skills = ["Communication","Leadership"]
technical_skills = [s.title() for s in skills_list if s.title() not in soft_skills]

# ---------------- FUNCTIONS ----------------
def extract_skills(text):
    detected = []
    text = text.lower()
    for skill in skills_list:
        if skill in text:
            detected.append(skill.title())
    return detected

def score_category(score):
    if score >= 75:
        return "Strong Profile"
    elif score >= 50:
        return "Moderate Profile"
    else:
        return "Needs Improvement"

def save_data(career, score, missing, confidence):
    file = "resume_data.csv"
    row = pd.DataFrame([{
        "Career": career,
        "Score": score,
        "Missing Skills": ", ".join(missing),
        "Confidence": confidence
    }])
    if os.path.exists(file):
        row.to_csv(file, mode="a", header=False, index=False)
    else:
        row.to_csv(file, index=False)

def generate_pdf(career_names, confidence_scores, detected_skills,
                 missing_skills, top_career, resume_score, graph_buffer):

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph("AI Career Intelligence Report", styles["Heading1"]))
    elements.append(Spacer(1, 15))

    elements.append(Paragraph("Top 3 Career Predictions", styles["Heading2"]))
    for name, score in zip(career_names, confidence_scores):
        elements.append(Paragraph(f"{name} - {score:.2f}%", styles["Normal"]))

    elements.append(Spacer(1, 15))
    elements.append(Paragraph(f"Resume Score: {resume_score}/100", styles["Heading2"]))
    elements.append(Paragraph(score_category(resume_score), styles["Normal"]))

    details = career_info.get(top_career)
    if details:
        elements.append(Spacer(1, 10))
        elements.append(Paragraph(f"Role: {top_career}", styles["Heading2"]))
        elements.append(Paragraph(details["description"], styles["Normal"]))
        elements.append(Paragraph(f"Salary: {details['salary']}", styles["Normal"]))

    elements.append(Spacer(1, 10))
    elements.append(Paragraph("Detected Skills", styles["Heading2"]))
    for skill in detected_skills:
        elements.append(Paragraph(f"- {skill}", styles["Normal"]))

    elements.append(Spacer(1, 10))
    elements.append(Paragraph("Missing Skills", styles["Heading2"]))
    for skill in missing_skills:
        elements.append(Paragraph(f"- {skill}", styles["Normal"]))

    elements.append(Spacer(1, 15))
    graph_buffer.seek(0)
    elements.append(Image(graph_buffer, width=400, height=250))

    doc.build(elements)
    buffer.seek(0)
    return buffer

@st.cache_resource
def load_models():
    model = pickle.load(open("career_model.pkl","rb"))
    vectorizer = pickle.load(open("vectorizer.pkl","rb"))
    encoder = pickle.load(open("career_encoder.pkl","rb"))
    return model, vectorizer, encoder

model, vectorizer, encoder = load_models()

# ---------------- NAVIGATION ----------------
menu = st.sidebar.radio("Navigation", 
                        ["🏠 Resume Analysis",
                         "📊 Analytics Dashboard",
                         "📈 Skill Gap Insights"])

# ================= PAGE 1 =================
if menu == "🏠 Resume Analysis":

    st.title("🚀 AI Career Intelligence System")

    uploaded_files = st.file_uploader(
        "Upload Resumes",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

    if st.button("Predict Career") and uploaded_files:

        summary_data = []

        for uploaded_file in uploaded_files:

            st.divider()
            st.subheader(f"📄 {uploaded_file.name}")

            resume_text = ""

            # -------- Extract Text --------
            if uploaded_file.type == "application/pdf":
                reader = PyPDF2.PdfReader(uploaded_file)
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        resume_text += text
            else:
                resume_text = uploaded_file.read().decode("utf-8")

            # -------- Prediction --------
            input_vector = vectorizer.transform([resume_text])
            probabilities = model.predict_proba(input_vector)[0]
            top_indices = np.argsort(probabilities)[-3:][::-1]

            career_names = []
            confidence_scores = []

            cols = st.columns(3)
            for i, index in enumerate(top_indices):
                name = encoder.inverse_transform([index])[0]
                confidence = probabilities[index] * 100
                career_names.append(name)
                confidence_scores.append(confidence)
                cols[i].metric(name, f"{confidence:.2f}%")

            # -------- Graph --------
            fig, ax = plt.subplots()
            ax.bar(career_names, confidence_scores)
            ax.set_title("Confidence Comparison")
            st.pyplot(fig)

            # -------- Skill Extraction --------
            detected = extract_skills(resume_text)

            resume_score = min(
                len(detected) * 10 + int(max(confidence_scores) // 2),
                100
            )

            top_career = career_names[0]

            # -------- Ranking --------
            if resume_score >= 85:
                rank = "🥇 Gold"
            elif resume_score >= 70:
                rank = "🥈 Silver"
            else:
                rank = "🥉 Bronze"

            st.subheader("📊 Resume Evaluation")
            st.metric("Score", f"{resume_score}/100")
            st.progress(resume_score)
            st.success(f"Rank Achieved: {rank}")

            required = career_info.get(top_career, {}).get("skills", [])
            missing = [s for s in required if s not in detected]

            # -------- Generate PDF --------
            from io import BytesIO
            graph_buffer = BytesIO()
            fig.savefig(graph_buffer, format="png")

            pdf_file = generate_pdf(
                career_names,
                confidence_scores,
                detected,
                missing,
                top_career,
                resume_score,
                graph_buffer
            )

            st.download_button(
                label="📥 Download Career Report",
                data=pdf_file,
                file_name=f"{uploaded_file.name}_career_report.pdf",
                mime="application/pdf"
            )

            save_data(
                top_career,
                resume_score,
                missing,
                max(confidence_scores)
            )

            summary_data.append({
                "Resume Name": uploaded_file.name,
                "Top Career": top_career,
                "Score": resume_score,
                "Confidence": round(max(confidence_scores), 2),
                "Rank": rank
            })

        # -------- Batch Ranking --------
        st.divider()
        st.header("🏆 Batch Ranking Summary")

        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values(by="Score", ascending=False)

        st.dataframe(summary_df, use_container_width=True)

        if not summary_df.empty:
            top_row = summary_df.iloc[0]
            st.success(
                f"🎉 Top Performer: {top_row['Resume Name']} "
                f"with Score {top_row['Score']}/100 ({top_row['Rank']})"
            )
# ================= PAGE 2 =================
if menu == "📊 Analytics Dashboard":

    st.title("📊 Analytics Dashboard")

    if os.path.exists("resume_data.csv"):

        df = pd.read_csv("resume_data.csv")

        # Show columns for debugging (optional)
        # st.write("Columns:", df.columns)

        # ---- SAFE COLUMN CLEANING ----
        df.columns = df.columns.str.strip()

        # ---- SAFE NUMERIC CONVERSION ----
        if "Score" in df.columns:
            df["Score"] = pd.to_numeric(df["Score"], errors="coerce")

        if "Confidence" in df.columns:
            df["Confidence"] = pd.to_numeric(df["Confidence"], errors="coerce")

        st.metric("Total Resumes", len(df))

        # ---- AVERAGE SCORE ----
        if "Score" in df.columns:
            avg_score = df["Score"].mean()
            if pd.notna(avg_score):
                st.metric("Average Score", round(avg_score, 2))
            else:
                st.metric("Average Score", "N/A")
        else:
            st.warning("Score column not found.")

        # ---------------- CAREER DISTRIBUTION ----------------
        st.subheader("Most Predicted Careers")

        if "Career" in df.columns:
            st.bar_chart(df["Career"].value_counts())
        else:
            st.error("Career column not found in dataset.")

        # ---------------- CONFIDENCE GRAPH ----------------
        st.subheader("Confidence Distribution")

        if "Confidence" in df.columns:
            st.line_chart(df["Confidence"])
        else:
            st.warning("Confidence column not found.")

        # ---------------- MISSING SKILLS ANALYSIS ----------------
        st.subheader("Most Missing Skills")

        if "Missing Skills" in df.columns:
            all_missing = df["Missing Skills"].dropna().astype(str).str.split(", ")
            exploded = all_missing.explode()

            if not exploded.empty:
                st.bar_chart(exploded.value_counts())
            else:
                st.info("No missing skills recorded yet.")
        else:
            st.warning("Missing Skills column not found.")

    else:
        st.info("No data available yet.")
# ================= PAGE 3 =================
if menu == "📈 Skill Gap Insights":

    st.title("📈 Skill Gap Insights")

    if os.path.exists("resume_data.csv"):
        df = pd.read_csv("resume_data.csv")

        if "Missing Skills" in df.columns:

            all_missing = df["Missing Skills"].dropna().astype(str).str.split(", ")
            exploded = all_missing.explode().dropna()

            if not exploded.empty:

                st.subheader("Skill Frequency")
                st.bar_chart(exploded.value_counts())

                # ---- SAFE SKILL CATEGORY COUNT ----
                exploded = exploded.astype(str)

                tech_count = sum(
                    skill.strip() in technical_skills
                    for skill in exploded
                )

                soft_count = sum(
                    skill.strip() in soft_skills
                    for skill in exploded
                )

                # ---- HANDLE EMPTY / ZERO CASE ----
                if tech_count == 0 and soft_count == 0:
                    st.info("No categorized skill data available.")
                else:
                    fig, ax = plt.subplots()
                    ax.pie(
                        [tech_count, soft_count],
                        labels=["Technical Skills", "Soft Skills"],
                        autopct="%1.1f%%"
                    )
                    ax.set_title("Skill Category Distribution")
                    st.pyplot(fig)

            else:
                st.info("No missing skills recorded yet.")

        else:
            st.info("Missing Skills column not found.")

    else:
        st.info("No skill data available.")
