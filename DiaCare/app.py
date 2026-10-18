# app.py — DiaCare (complete, Pinecone RAG + DeepSeek + ML + Report)

import os
import json
import logging
import sqlite3
import re
from datetime import datetime, date
from io import BytesIO

from flask import (
    Flask, render_template, request, redirect, url_for,
    session, flash, Response, send_from_directory, abort,
    send_file, get_flashed_messages, jsonify
)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import pdfplumber

from dotenv import load_dotenv

# Load environment variables from api.env
load_dotenv("api.env")

# ============= CONFIG =============
BASE_DIR = os.path.dirname(__file__)
DB_PATH = os.path.join(BASE_DIR, 'diacare.db')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'txt'}

# ── Single Flask app instance ──────────────────────────────────────────────────
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load secret key
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "fallback-dev-secret-key")

# Load API keys
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")

# Logo path (use relative path so it works on any machine)
LOGO_PATH = os.path.join(BASE_DIR, "images", "diacare_logo.png")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("diacare")
logging.getLogger("werkzeug").setLevel(logging.ERROR)

# ============= Optional imports =============
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    from enhanced_rag_connector import enhanced_rag
except Exception:
    enhanced_rag = None

# ============= Helpers =============
def calculate_age(dob_str):
    if not dob_str:
        return "Unknown"
    try:
        dob = datetime.strptime(dob_str, "%Y-%m-%d").date()
        today = date.today()
        return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
    except Exception:
        return "Unknown"


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ============= Database Initialization ============
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Patients
    c.execute('''CREATE TABLE IF NOT EXISTS patients
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  full_name TEXT NOT NULL,
                  email TEXT UNIQUE NOT NULL,
                  password TEXT NOT NULL,
                  phone TEXT,
                  date_of_birth TEXT,
                  gender TEXT,
                  address TEXT,
                  emergency_contact TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')

    # Doctors
    c.execute('''CREATE TABLE IF NOT EXISTS doctors
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  full_name TEXT NOT NULL,
                  email TEXT UNIQUE NOT NULL,
                  password TEXT NOT NULL,
                  phone TEXT,
                  specialization TEXT,
                  license_number TEXT,
                  hospital_name TEXT,
                  years_experience INTEGER,
                  address TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')

    # Predictions
    c.execute('''CREATE TABLE IF NOT EXISTS predictions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  patient_id INTEGER NOT NULL,
                  pregnancies INTEGER,
                  glucose REAL,
                  blood_pressure REAL,
                  skin_thickness REAL,
                  insulin REAL,
                  bmi REAL,
                  diabetes_pedigree REAL,
                  age INTEGER,
                  systolic_bp REAL,
                  hba1c REAL,
                  triglycerides REAL,
                  prediction_result TEXT,
                  risk_probability REAL,
                  recommendations TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (patient_id) REFERENCES patients (id))''')

    # Lab reports
    c.execute('''CREATE TABLE IF NOT EXISTS lab_reports
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  patient_id INTEGER NOT NULL,
                  report_type TEXT,
                  file_name TEXT,
                  file_path TEXT,
                  notes TEXT,
                  uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (patient_id) REFERENCES patients (id))''')

    # Doctor-patient link
    c.execute('''CREATE TABLE IF NOT EXISTS doctor_patients
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  doctor_id INTEGER NOT NULL,
                  patient_id INTEGER NOT NULL,
                  assigned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (doctor_id) REFERENCES doctors (id),
                  FOREIGN KEY (patient_id) REFERENCES patients (id))''')

    # Appointments
    c.execute('''CREATE TABLE IF NOT EXISTS appointments
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  patient_id INTEGER NOT NULL,
                  doctor_id INTEGER NOT NULL,
                  appointment_date TEXT,
                  appointment_time TEXT,
                  reason TEXT,
                  status TEXT DEFAULT 'Pending',
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (patient_id) REFERENCES patients(id),
                  FOREIGN KEY (doctor_id) REFERENCES doctors(id))''')

    conn.commit()
    conn.close()
    logger.info("Database initialized")


init_db()


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ============= RAG + LLM Initialization ============
retrieve_sections = None
deepseek = None

if enhanced_rag:
    try:
        rag_cfg = enhanced_rag()
        if rag_cfg.get("status") == "active":
            retrieve_sections = rag_cfg["retrieve_fn"]
            logger.info("RAG ready – Pinecone connected")
        else:
            logger.warning("RAG unavailable: %s", rag_cfg.get("reason"))
    except Exception as e:
        retrieve_sections = None
        logger.exception("RAG init failed: %s", e)
else:
    logger.warning("enhanced_rag_connector not found or failed to import.")

if OpenAI:
    try:
        deepseek = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url=OPENAI_API_BASE
        )
        print("✅ OpenRouter client initialized successfully")
        logger.info("OpenRouter client configured")
    except Exception as e:
        deepseek = None
        print(f"❌ OpenRouter client init failed: {e}")
else:
    print("❌ openai package missing — run: pip install openai")


# ============= ML Model =============
import joblib
import pandas as pd

MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "diabetes_prediction.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")

print("\n================ MODEL LOAD DEBUG ================")
print("📌 Model path =", MODEL_PATH)
print("📌 Scaler path =", SCALER_PATH)
print("📌 Model exists:", os.path.exists(MODEL_PATH))
print("📌 Scaler exists:", os.path.exists(SCALER_PATH))

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

print("✅ 3-CLASS MODEL + SCALER LOADED SUCCESSFULLY!")
print("==================================================\n")

try:
    ct = model.named_steps['preprocess']
    ohe = ct.named_transformers_['cat']
    print("\n========== ENCODER DEBUG ==========")
    print("Encoder categories for gender:", ohe.categories_)
    print("====================================\n")
except Exception as e:
    print("ERROR READING ENCODER:", e)


# ============= Report Generator =============
def generate_diabetes_report(input_data, prediction, probability):
    patient_summary = f"""
Patient Diabetes Assessment Summary
-----------------------------------
Gender: {"Male" if input_data['gender'] == 1 else "Female"}
Age: {input_data['age']}
BMI: {input_data['bmi']}
Systolic Blood Pressure: {input_data['systolic_bp']} mmHg
Glucose Level: {input_data['glucose']} mg/dL
HbA1c: {input_data['hba1c']}%
Triglycerides: {input_data['triglycerides']} mg/dL

Model Prediction: {prediction}
Risk Probability: {round(probability * 100, 2)}%
"""

    try:
        if retrieve_sections:
            kb_sections = retrieve_sections(f"Diabetes assessment:\n{patient_summary}")
            rag_context = "\n\n".join(kb_sections) if kb_sections else "No knowledge-base context found."
        else:
            rag_context = "RAG unavailable."
    except Exception as e:
        rag_context = f"RAG retrieval error: {e}"

    final_prompt = f"""
You are a senior diabetologist and medical report specialist.
Generate a premium, clinically polished diabetes assessment report.

IMPORTANT OUTPUT RULES (STRICT)
-------------------------------------------------------------
• Output ONLY the report content.
• DO NOT add any introduction or meta text (e.g., "Here is your report").
• DO NOT add explanations of what you are doing.
• DO NOT use Markdown or code fences.
• DO NOT include <html>, <head>, <body>, <style>, CSS, or scripts.
• USE ONLY these HTML tags: <h2>, <h3>, <p>, <b>, <ul>, <li>, <hr>, <br>.
• Emojis ARE allowed (⭐, ⚠️, ❗).
• Headings must appear clean and premium — NO inline styles.
• Bold important medical keywords or findings.
• Make paragraphs short, crisp, and clinically meaningful.

-------------------------------------------------------------
SECTIONS TO GENERATE (MUST FOLLOW ORDER)
-------------------------------------------------------------

<h2>EXECUTIVE SUMMARY</h2>
Write a concise medical overview:
• Interpret the diabetes classification
• Explain the meaning of the risk probability
• Highlight the dominant metabolic factors
Tone: professional, reassuring, premium.

<hr>

<h2>RISK BREAKDOWN</h2>
Present findings as a neat bullet list:
• Glucose — Normal ⭐ / Borderline ⚠️ / High ❗
• HbA1c — with tag
• BMI — with interpretation
• Blood Pressure — metabolic relevance
• Triglycerides — cardiovascular relevance
• Age-related risk category

Tags allowed:
Normal ⭐
Borderline ⚠️
High ❗

<hr>

<h2>CLINICAL INTERPRETATION</h2>
Explain physiological meaning:
• Insulin sensitivity
• Glycemic control stability
• Lipid-driven metabolic strain
• Long-term implications (without fear language)

<hr>

<h2>LIFESTYLE GUIDANCE</h2>
Provide general, non-medication guidance:
<b>Diet:</b> Recommended patterns
<b>Exercise:</b> Intensity & frequency
<b>Sleep & Hydration:</b> Best practices
<b>Behavior:</b> Stress, habits, consistency

<hr>

<h2>FOLLOW-UP PLAN</h2>
• When to repeat tests
• What biomarkers to monitor
• When clinical review is advised

<hr>

<h2>WARNING SIGNS</h2>
Use <ul> with ⚠️ icons.
Include symptoms that require early medical evaluation.

<hr>

<h2>CLINICAL SUMMARY</h2>
A polished closing paragraph summarizing:
• Patient's current metabolic condition
• Forward steps
• Reassurance and clarity

-------------------------------------------------------------
PATIENT DATA (REFERENCE FOR REPORT)
-------------------------------------------------------------
{patient_summary}

Generate the final formatted report NOW.
Remember: output ONLY the report.
"""

    if not deepseek:
        return "(LLM unavailable)\n\n" + patient_summary

    try:
        resp = deepseek.chat.completions.create(
            model="meta-llama/llama-3.3-70b-instruct",
            messages=[{"role": "user", "content": final_prompt}],
            temperature=0.3,
            max_tokens=900
        )
        try:
            answer = resp.choices[0].message.content
        except Exception:
            answer = str(resp)
        return answer.strip()
    except Exception as e:
        return f"LLM generation failed: {e}"


# ============= PDF helpers =============
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.lib.utils import ImageReader

FONT_DIR = os.path.join(BASE_DIR, "fonts")
pdfmetrics.registerFont(TTFont("Poppins", os.path.join(FONT_DIR, "Poppins-Regular.ttf")))
pdfmetrics.registerFont(TTFont("Poppins-SemiBold", os.path.join(FONT_DIR, "Poppins-SemiBold.ttf")))
pdfmetrics.registerFont(TTFont("Poppins-Bold", os.path.join(FONT_DIR, "Poppins-Bold.ttf")))
registerFontFamily(
    "Poppins",
    normal="Poppins",
    bold="Poppins-Bold",
    italic="Poppins",
    boldItalic="Poppins-Bold"
)


def colorize_risk_terms(text):
    text = re.sub(r"Normal\s*⭐",
                  "<font color='#16A34A'><b>Normal ⭐</b></font>", text, flags=re.IGNORECASE)
    text = re.sub(r"Borderline\s*⚠️",
                  "<font color='#F59E0B'><b>Borderline ⚠️</b></font>", text, flags=re.IGNORECASE)
    text = re.sub(r"High\s*❗",
                  "<font color='#DC2626'><b>High ❗</b></font>", text, flags=re.IGNORECASE)
    return text


def add_header_footer(canvas, doc):
    """Draws header with logo + title and watermark on every page."""
    canvas.saveState()
    PAGE_W, PAGE_H = A4

    # Header logo
    try:
        logo = ImageReader(LOGO_PATH)
        LOGO_W, LOGO_H = 60, 60
        LOGO_X = 45
        LOGO_Y = PAGE_H - 70
        canvas.drawImage(logo, LOGO_X, LOGO_Y, width=LOGO_W, height=LOGO_H,
                         preserveAspectRatio=True, mask='auto')
    except Exception as e:
        print("❌ HEADER LOGO ERROR:", e)

    # Header title
    TITLE_X = 45 + 60 + 20
    TITLE_Y = PAGE_H - 45
    canvas.setFont("Poppins-Bold", 22)
    canvas.setFillColor(colors.HexColor("#667eea"))
    canvas.drawString(TITLE_X, TITLE_Y, "DiaCare AI Clinical Report")

    # Divider line below header
    LINE_Y = PAGE_H - 80
    canvas.setStrokeColor(colors.HexColor("#d1d5db"))
    canvas.setLineWidth(1)
    canvas.line(45, LINE_Y, PAGE_W - 45, LINE_Y)

    # Watermark
    try:
        wm = ImageReader(LOGO_PATH)
        canvas.saveState()
        canvas.setFillAlpha(0.08)
        canvas.setStrokeAlpha(0.08)
        WM_W, WM_H = 350, 350
        canvas.drawImage(wm, (PAGE_W - WM_W) / 2, (PAGE_H - WM_H) / 2,
                         width=WM_W, height=WM_H, preserveAspectRatio=True, mask='auto')
        canvas.restoreState()
    except Exception as e:
        print("❌ WATERMARK ERROR:", e)

    # Footer
    canvas.setFont("Poppins", 9)
    canvas.setFillColor(colors.grey)
    footer_text = "Report generated by DiaCare AI System"
    text_width = canvas.stringWidth(footer_text, "Poppins", 9)
    canvas.drawString((PAGE_W - text_width) / 2, 25, footer_text)

    canvas.restoreState()


def render_html_section_to_pdf(html, story, styles, PRIMARY):
    html = html.replace("<br>", "\n").replace("<br/>", "\n")
    blocks = re.split(r"<hr\s*/?>", html)

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        # H2 Headers
        headers = re.findall(r"<h2>(.*?)</h2>", block)
        for header in headers:
            emoji = ""
            if "EXECUTIVE" in header.upper(): emoji = " ⭐"
            if "RISK" in header.upper(): emoji = " ⚠️"
            if "WARNING" in header.upper(): emoji = " ❗"
            story.append(Paragraph(header + emoji, styles["dc_section"]))
            story.append(HRFlowable(width="100%", color=colors.HexColor("#e5e7eb"), thickness=0.8))
            story.append(Spacer(1, 6))
            block = block.replace(f"<h2>{header}</h2>", "")

        # H3 Subheaders
        subs = re.findall(r"<h3>(.*?)</h3>", block)
        for sub in subs:
            story.append(Paragraph(sub, styles["dc_subheader"]))
            story.append(Spacer(1, 4))
            block = block.replace(f"<h3>{sub}</h3>", "")

        # Bullet lists
        ul_match = re.search(r"<ul>(.*?)</ul>", block, flags=re.DOTALL)
        if ul_match:
            ul_content = ul_match.group(1)
            lis = re.findall(r"<li>(.*?)</li>", ul_content, flags=re.DOTALL)
            for li in lis:
                li_clean = li.strip()
                bullet_text = li_clean if li_clean.startswith(("⚠️", "❗", "⭐")) else "• " + li_clean
                story.append(Paragraph(bullet_text, styles["dc_normal"]))
                story.append(Spacer(1, 5))
            block = block.replace(ul_match.group(0), "")
            continue

        # Paragraphs
        paragraphs = re.findall(r"<p>(.*?)</p>", block)
        for para in paragraphs:
            story.append(Paragraph(para, styles["dc_normal"]))
            story.append(Spacer(1, 8))


# ============= Lab extraction helpers =============
TEMP_EXTRACT_DIR = "/tmp/diacare_extract"
os.makedirs(TEMP_EXTRACT_DIR, exist_ok=True)

EXTRACTION_PROMPT = """
You are an expert medical data extraction AI. Your task is to extract ONLY specific medical values from the provided report text.

REQUIRED VALUES TO EXTRACT:
1. **Glucose** - Blood sugar level (mg/dL)
2. **HbA1c** - Glycated hemoglobin (%)
3. **Triglycerides** - Lipid level (mg/dL)
4. **Systolic_BP** - Systolic blood pressure (mmHg)
5. **BMI** - Body Mass Index

EXTRACTION RULES:
- Extract ONLY the patient's actual test result values
- IGNORE reference ranges, normal ranges, and lab standards
- For Blood Pressure: Extract ONLY the systolic (first/top) number
- Return null if a value is not found or unclear

OUTPUT FORMAT (CRITICAL):
Return ONLY valid JSON with this exact structure. NO markdown, NO code fences, NO explanations.

{{
  "glucose": <number or null>,
  "hba1c": <number or null>,
  "triglycerides": <number or null>,
  "systolic_bp": <number or null>,
  "bmi": <number or null>
}}

REPORT TEXT:
{text}
"""


def extract_text_from_file(file):
    if not file or file.filename == '':
        return ""
    if file.filename.lower().endswith('.pdf'):
        try:
            text_chunks = []
            file_bytes = file.read()
            with pdfplumber.open(BytesIO(file_bytes)) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text()
                    if page_text:
                        text_chunks.append(f"--- Page {page_num} ---\n{page_text}")
            full_text = "\n\n".join(text_chunks)
            return full_text.strip()
        except Exception as e:
            print(f"❌ PDF extraction error: {e}")
            return ""
    return ""


def extract_medical_values_llm(text):
    if not text or len(text.strip()) < 10:
        return {}
    if not deepseek:
        return {}
    if len(text) > 6000:
        text = text[:6000] + "\n... [truncated]"

    prompt = EXTRACTION_PROMPT.format(text=text)
    try:
        response = deepseek.chat.completions.create(
            model="deepseek/deepseek-chat",
            messages=[
                {"role": "system",
                 "content": "You are a medical data extraction specialist. Return only valid JSON with no additional text."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=300
        )
        raw_response = response.choices[0].message.content.strip()
        cleaned = raw_response.replace("```json", "").replace("```", "").strip()
        extracted_data = json.loads(cleaned)
        valid_keys = {"glucose", "hba1c", "triglycerides", "systolic_bp", "bmi"}
        return {k: v for k, v in extracted_data.items() if k in valid_keys}
    except Exception as e:
        print(f"❌ LLM Extraction Error: {e}")
        return {}


def merge_extracted_data(existing_data, new_data):
    merged = existing_data.copy() if existing_data else {}
    for key, value in new_data.items():
        if value is not None:
            merged[key] = value
    return merged


# ============= Bleach / Markup =============
from markupsafe import Markup
import bleach

ALLOWED_TAGS = ["h2", "h3", "p", "b", "ul", "li", "hr", "br"]


# ============= Routes =============

@app.route('/')
def home():
    return render_template('home.html')


# ── Patient register/login ─────────────────────────────────────────────────────

@app.route('/patient_register', methods=['GET', 'POST'])
@app.route('/patient/register', methods=['GET', 'POST'])
def patient_register():
    if request.method == 'GET':
        return render_template('patient_register.html')

    full_name = request.form.get('full_name', '').strip()
    email = request.form.get('email', '').strip().lower()
    password = request.form.get('password', '')
    phone = request.form.get('phone', '')
    dob = request.form.get('date_of_birth', '')
    gender = request.form.get('gender', '')
    address = request.form.get('address', '')
    emergency = request.form.get('emergency_contact', '')

    if not full_name or not email or not password:
        flash("Name, email and password required", "danger")
        return redirect(url_for('patient_register'))

    pw_hash = generate_password_hash(password)
    conn = get_db()
    c = conn.cursor()
    try:
        c.execute('''INSERT INTO patients
                     (full_name, email, password, phone, date_of_birth, gender, address, emergency_contact)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                  (full_name, email, pw_hash, phone, dob, gender, address, emergency))
        conn.commit()
        flash("Patient registered — please log in", "success")
        return redirect(url_for('patient_login'))
    except sqlite3.IntegrityError:
        flash("A patient with that email already exists.", "danger")
        return redirect(url_for('patient_register'))
    finally:
        conn.close()


@app.route('/patient_login', methods=['GET', 'POST'])
@app.route('/patient/login', methods=['GET', 'POST'])
def patient_login():
    if request.method == 'GET':
        get_flashed_messages()  # clear stale messages
        return render_template('patient_login.html')

    email = request.form.get('email', '').strip().lower()
    password = request.form.get('password', '')

    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM patients WHERE email = ?", (email,))
    user = c.fetchone()
    conn.close()

    if not user:
        flash("No patient account with that email.", "danger")
        return redirect(url_for('patient_login'))

    if not check_password_hash(user['password'], password):
        flash("Incorrect password.", "danger")
        return redirect(url_for('patient_login'))

    session.clear()
    session['user_id'] = user['id']
    session['user_type'] = 'patient'
    session['user_name'] = user['full_name']
    flash("Logged in as patient", "success")
    return redirect(url_for('dashboard'))


@app.route("/patient/forgot-password", methods=["POST"])
def patient_forgot_password():
    email = request.form.get("email", "").strip()
    dob = request.form.get("dob", "").strip()
    new = request.form.get("new_password", "")
    confirm = request.form.get("confirm_password", "")

    if new != confirm:
        flash("Passwords do not match.", "danger")
        return redirect(url_for("patient_login"))

    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT id FROM patients WHERE email = ? AND date_of_birth = ?", (email, dob))
    user = c.fetchone()

    if not user:
        flash("Email and DOB do not match our records.", "danger")
        conn.close()
        return redirect(url_for("patient_login"))

    hashed = generate_password_hash(new)
    c.execute("UPDATE patients SET password = ? WHERE email = ?", (hashed, email))
    conn.commit()
    conn.close()

    flash("Password updated successfully. Please login.", "success")
    return redirect(url_for("patient_login"))


# ── Doctor register/login ──────────────────────────────────────────────────────

@app.route('/doctor_register', methods=['GET', 'POST'])
@app.route('/doctor/register', methods=['GET', 'POST'])
def doctor_register():
    if request.method == 'GET':
        return render_template('doctor_register.html')

    full_name = request.form.get('full_name', '').strip()
    email = request.form.get('email', '').strip().lower()
    password = request.form.get('password', '')
    phone = request.form.get('phone', '')
    specialization = request.form.get('specialization', '')
    license_num = request.form.get('license_number', '')
    hospital = request.form.get('hospital_name', '')
    years_exp = request.form.get('years_experience') or 0

    if not full_name or not email or not password:
        flash("Name, email and password required", "danger")
        return redirect(url_for('doctor_register'))

    pw_hash = generate_password_hash(password)
    conn = get_db()
    c = conn.cursor()
    try:
        c.execute('''INSERT INTO doctors
                     (full_name, email, password, phone, specialization, license_number, hospital_name, years_experience)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                  (full_name, email, pw_hash, phone, specialization, license_num, hospital, years_exp))
        conn.commit()
        flash("Doctor registered — please log in", "success")
        return redirect(url_for('doctor_login'))
    except sqlite3.IntegrityError:
        flash("A doctor with that email already exists.", "danger")
        return redirect(url_for('doctor_register'))
    finally:
        conn.close()


@app.route("/doctor_login", methods=["GET", "POST"])
@app.route('/doctor/login', methods=['GET', 'POST'])
def doctor_login():
    if request.method == 'GET':
        get_flashed_messages()
        return render_template('doctor_login.html')

    email = request.form.get('email', '').strip().lower()
    password = request.form.get('password', '')

    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM doctors WHERE email = ?", (email,))
    doctor = c.fetchone()
    conn.close()

    if not doctor:
        flash("No doctor account with that email.", "danger")
        return redirect(url_for('doctor_login'))

    if not check_password_hash(doctor['password'], password):
        flash("Incorrect password.", "danger")
        return redirect(url_for('doctor_login'))

    session.clear()
    session['user_id'] = doctor['id']
    session['user_name'] = doctor['full_name']
    session['user_type'] = 'doctor'
    return redirect(url_for('doctor_dashboard'))


@app.route("/doctor/forgot-password", methods=["POST"])
def doctor_forgot_password():
    email = request.form.get("email", "").strip()
    license_no = request.form.get("license_number", "").strip()
    new = request.form.get("new_password", "").strip()
    confirm = request.form.get("confirm_password", "").strip()

    if not all([email, license_no, new, confirm]):
        flash("All fields are required.", "danger")
        return redirect(url_for("doctor_login"))

    if new != confirm:
        flash("Confirm password does not match.", "danger")
        return redirect(url_for("doctor_login"))

    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT id FROM doctors WHERE email = ? AND license_number = ?", (email, license_no))
    doctor = c.fetchone()

    if not doctor:
        flash("Details do not match our records.", "danger")
        conn.close()
        return redirect(url_for("doctor_login"))

    hashed = generate_password_hash(new)
    c.execute("UPDATE doctors SET password = ? WHERE email = ?", (hashed, email))
    conn.commit()
    conn.close()

    flash("Password updated successfully. Please login.", "success")
    return redirect(url_for("doctor_login"))


# ── Logout ─────────────────────────────────────────────────────────────────────

@app.route('/logout')
def logout():
    session.clear()
    flash("Logged out", "info")
    return redirect(url_for('home'))


# ── Patient dashboard ──────────────────────────────────────────────────────────

@app.route('/dashboard')
def dashboard():
    if session.get('user_type') != 'patient':
        return redirect(url_for('patient_login'))

    patient_id = session['user_id']
    conn = get_db()
    c = conn.cursor()

    c.execute("SELECT * FROM patients WHERE id = ?", (patient_id,))
    patient = c.fetchone()

    c.execute('''SELECT d.* FROM doctors d
                 JOIN doctor_patients dp ON dp.doctor_id = d.id
                 WHERE dp.patient_id = ?''', (patient_id,))
    doctors = c.fetchall()

    c.execute("""SELECT * FROM predictions
                 WHERE patient_id = ?
                 ORDER BY created_at DESC LIMIT 5""", (patient_id,))
    predictions = c.fetchall()

    c.execute("""SELECT * FROM lab_reports
                 WHERE patient_id = ?
                 ORDER BY uploaded_at DESC""", (patient_id,))
    lab_reports = c.fetchall()

    conn.close()

    return render_template(
        'dashboard.html',
        patient=patient,
        doctors=doctors,
        predictions=predictions,
        lab_reports=lab_reports
    )


@app.route('/remove_doctor', methods=['POST'])
def remove_doctor():
    if session.get('user_type') != 'patient':
        return redirect(url_for('patient_login'))

    patient_id = session.get('user_id')
    doctor_id = request.form.get('doctor_id')

    conn = get_db()
    c = conn.cursor()
    c.execute("DELETE FROM doctor_patients WHERE doctor_id = ? AND patient_id = ?",
              (doctor_id, patient_id))
    conn.commit()
    conn.close()

    flash("Doctor removed successfully.", "info")
    return redirect(url_for('dashboard'))


# ── Patient settings ───────────────────────────────────────────────────────────

@app.route('/patient/settings')
def patient_settings():
    if session.get('user_type') != 'patient':
        return redirect(url_for('patient_login'))

    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM patients WHERE id = ?", (session['user_id'],))
    patient = c.fetchone()
    conn.close()

    if not patient:
        flash("Patient data not found.", "danger")
        return redirect(url_for('patient_login'))

    return render_template('patient_settings.html', patient=patient)


@app.route('/patient/settings/update', methods=['POST'])
def patient_settings_update():
    if session.get('user_type') != 'patient':
        return redirect(url_for('patient_login'))

    user_id = session.get('user_id')
    full_name = request.form.get('full_name')
    email = request.form.get('email')
    phone = request.form.get('phone')
    date_of_birth = request.form.get('date_of_birth')
    gender = request.form.get('gender')
    address = request.form.get('address')

    conn = get_db()
    c = conn.cursor()
    c.execute("""UPDATE patients
                 SET full_name=?, email=?, phone=?, date_of_birth=?, gender=?, address=?
                 WHERE id=?""",
              (full_name, email, phone, date_of_birth, gender, address, user_id))
    conn.commit()
    conn.close()

    flash("Profile updated successfully!", "success")
    return redirect(url_for('patient_settings'))


@app.route("/patient/settings/change-password", methods=["POST"])
def patient_change_password():
    if session.get("user_type") != "patient":
        return redirect(url_for("patient_login"))

    user_id = session.get("user_id")
    current_password = request.form.get("current_password")
    new_password = request.form.get("new_password")
    confirm_password = request.form.get("confirm_password")

    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM patients WHERE id = ?", (user_id,))
    patient = c.fetchone()

    if not patient:
        flash("Patient not found.", "danger")
        conn.close()
        return redirect(url_for("patient_settings"))

    if not check_password_hash(patient["password"], current_password):
        flash("Incorrect current password.", "danger")
        conn.close()
        return redirect(url_for("patient_settings"))

    if new_password != confirm_password:
        flash("New passwords do not match.", "danger")
        conn.close()
        return redirect(url_for("patient_settings"))

    hashed_pw = generate_password_hash(new_password)
    c.execute("UPDATE patients SET password = ? WHERE id = ?", (hashed_pw, user_id))
    conn.commit()
    conn.close()

    flash("Password updated successfully.", "success")
    return redirect(url_for("patient_settings"))


@app.route('/patient/settings/delete', methods=['POST'])
def patient_settings_delete():
    if session.get('user_type') != 'patient':
        return redirect(url_for('patient_login'))

    user_id = session.get('user_id')
    conn = get_db()
    c = conn.cursor()
    c.execute("DELETE FROM patients WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()

    session.clear()
    flash("Account deleted.", "info")
    return redirect(url_for('home'))


# ── Doctor dashboard ───────────────────────────────────────────────────────────

@app.route('/doctor/dashboard')
def doctor_dashboard():
    if session.get('user_type') != 'doctor':
        return redirect(url_for('doctor_login'))

    doctor_id = session['user_id']
    conn = get_db()
    c = conn.cursor()

    c.execute("SELECT * FROM doctors WHERE id = ?", (doctor_id,))
    doctor = c.fetchone()

    c.execute("SELECT COUNT(*) FROM doctor_patients WHERE doctor_id = ?", (doctor_id,))
    patients_count = c.fetchone()[0] or 0

    c.execute("""SELECT a.id, a.appointment_date, a.appointment_time, a.reason, a.status,
                        p.full_name AS patient_name
                 FROM appointments a
                 JOIN patients p ON p.id = a.patient_id
                 WHERE a.doctor_id = ?
                 ORDER BY a.created_at DESC""", (doctor_id,))
    appointments = c.fetchall()

    pending_appointments_count = sum(1 for a in appointments if a["status"] == "Pending")

    c.execute("""SELECT COUNT(DISTINCT patient_id) FROM predictions
                 WHERE prediction_result = 'Diabetes'
                   AND patient_id IN (SELECT patient_id FROM doctor_patients WHERE doctor_id = ?)""",
              (doctor_id,))
    high_risk_count = c.fetchone()[0] or 0

    c.execute("""SELECT p.full_name AS patient_name, pr.*
                 FROM predictions pr
                 JOIN patients p ON p.id = pr.patient_id
                 WHERE pr.patient_id IN (SELECT patient_id FROM doctor_patients WHERE doctor_id = ?)
                 ORDER BY pr.created_at DESC LIMIT 5""", (doctor_id,))
    recent_predictions = c.fetchall()

    conn.close()

    return render_template(
        'doctor_dashboard.html',
        doctor=doctor,
        patients_count=patients_count,
        appointments=appointments,
        pending_appointments_count=pending_appointments_count,
        high_risk_count=high_risk_count,
        recent_predictions=recent_predictions
    )


@app.route('/doctor/profile')
def doctor_profile():
    if session.get('user_type') != 'doctor':
        return redirect(url_for('doctor_login'))

    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM doctors WHERE id = ?", (session['user_id'],))
    doctor = c.fetchone()
    conn.close()
    return render_template('doctor_profile.html', doctor=doctor)


@app.route("/doctor/patients")
def view_patients():
    if session.get("user_type") != "doctor":
        return redirect(url_for("doctor_login"))

    doctor_id = session["user_id"]
    conn = get_db()
    c = conn.cursor()

    c.execute("""SELECT p.* FROM patients p
                 JOIN doctor_patients dp ON dp.patient_id = p.id
                 WHERE dp.doctor_id = ?""", (doctor_id,))
    patients = c.fetchall()

    latest_data = {}
    for p in patients:
        c.execute("""SELECT * FROM predictions WHERE patient_id = ?
                     ORDER BY created_at DESC LIMIT 1""", (p["id"],))
        latest_data[p["id"]] = c.fetchone()

    conn.close()
    return render_template("doctor_patients.html", patients=patients, latest_data=latest_data)


@app.route('/doctor/analytics')
def doctor_analytics():
    if session.get('user_type') != 'doctor':
        return redirect(url_for('doctor_login'))

    doctor_id = session['user_id']
    conn = get_db()
    c = conn.cursor()

    c.execute("SELECT COUNT(*) FROM doctor_patients WHERE doctor_id = ?", (doctor_id,))
    patients_count = c.fetchone()[0] or 0

    c.execute("""SELECT COUNT(*) FROM predictions
                 WHERE patient_id IN (SELECT patient_id FROM doctor_patients WHERE doctor_id = ?)""",
              (doctor_id,))
    predictions_count = c.fetchone()[0] or 0

    c.execute("""SELECT
                   SUM(CASE WHEN prediction_result = 'Normal' THEN 1 ELSE 0 END),
                   SUM(CASE WHEN prediction_result = 'Prediabetes' THEN 1 ELSE 0 END),
                   SUM(CASE WHEN prediction_result = 'Diabetes' THEN 1 ELSE 0 END)
                 FROM predictions
                 WHERE patient_id IN (SELECT patient_id FROM doctor_patients WHERE doctor_id = ?)""",
              (doctor_id,))
    low, moderate, high = c.fetchone()
    low = low or 0
    moderate = moderate or 0
    high = high or 0
    total = max(low + moderate + high, 1)
    low_risk_percentage = round((low / total) * 100, 2)
    moderate_risk_percentage = round((moderate / total) * 100, 2)
    high_risk_percentage = round((high / total) * 100, 2)

    c.execute("""SELECT COUNT(*) FROM predictions
                 WHERE created_at >= datetime('now', '-7 days')
                   AND patient_id IN (SELECT patient_id FROM doctor_patients WHERE doctor_id = ?)""",
              (doctor_id,))
    weekly_count = c.fetchone()[0] or 0

    c.execute("""SELECT strftime('%Y-%m', created_at) AS month, COUNT(*)
                 FROM predictions
                 WHERE patient_id IN (SELECT patient_id FROM doctor_patients WHERE doctor_id = ?)
                 GROUP BY month ORDER BY month""", (doctor_id,))
    monthly_data = c.fetchall()
    months = [row[0] for row in monthly_data]
    monthly_counts = [row[1] for row in monthly_data]

    c.execute("""SELECT age FROM predictions
                 WHERE patient_id IN (SELECT patient_id FROM doctor_patients WHERE doctor_id = ?)""",
              (doctor_id,))
    ages = [row[0] for row in c.fetchall() if row[0] is not None]
    age_groups = {"0-20": 0, "21-40": 0, "41-60": 0, "61+": 0}
    for age in ages:
        if age <= 20:
            age_groups["0-20"] += 1
        elif age <= 40:
            age_groups["21-40"] += 1
        elif age <= 60:
            age_groups["41-60"] += 1
        else:
            age_groups["61+"] += 1

    c.execute("""SELECT strftime('%Y-%m', created_at) AS month, AVG(bmi)
                 FROM predictions
                 WHERE patient_id IN (SELECT patient_id FROM doctor_patients WHERE doctor_id = ?)
                 GROUP BY month ORDER BY month""", (doctor_id,))
    bmi_results = c.fetchall()
    bmi_months = [row[0] for row in bmi_results]
    bmi_values = [round(row[1], 2) for row in bmi_results]

    conn.close()

    return render_template(
        "doctor_analytics.html",
        patients_count=patients_count,
        predictions_count=predictions_count,
        low_risk_percentage=low_risk_percentage,
        moderate_risk_percentage=moderate_risk_percentage,
        high_risk_percentage=high_risk_percentage,
        low_risk_count=low,
        moderate_risk_count=moderate,
        high_risk_count=high,
        weekly_count=weekly_count,
        months=months,
        monthly_counts=monthly_counts,
        age_groups=age_groups,
        bmi_months=bmi_months,
        bmi_values=bmi_values
    )


@app.route('/doctor/patient/<int:patient_id>')
def doctor_view_patient(patient_id):
    if session.get('user_type') != 'doctor':
        return redirect(url_for('doctor_login'))

    conn = get_db()
    c = conn.cursor()

    c.execute("SELECT * FROM patients WHERE id = ?", (patient_id,))
    patient = c.fetchone()

    if not patient:
        flash("Patient not found.", "danger")
        conn.close()
        return redirect(url_for('view_patients'))

    age = calculate_age(patient["date_of_birth"])

    c.execute("""SELECT * FROM predictions WHERE patient_id = ?
                 ORDER BY created_at DESC""", (patient_id,))
    rows = c.fetchall()
    conn.close()

    assessments = []
    for a in rows:
        a = dict(a)
        if a.get("recommendations"):
            clean_html = bleach.clean(a["recommendations"], tags=ALLOWED_TAGS, strip=True)
            a["recommendations"] = Markup(clean_html)
        assessments.append(a)

    return render_template("doctor_patient_view.html", patient=patient, assessments=assessments, age=age)


# ── Doctor settings ────────────────────────────────────────────────────────────

@app.route('/doctor/settings')
def doctor_settings():
    if session.get('user_type') != 'doctor':
        return redirect(url_for('doctor_login'))

    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM doctors WHERE id = ?", (session['user_id'],))
    doctor = c.fetchone()
    conn.close()

    return render_template('doctor_settings.html', doctor=doctor)


@app.route('/doctor/settings/update', methods=['POST'])
def doctor_settings_update():
    if session.get('user_type') != 'doctor':
        return redirect(url_for('doctor_login'))

    doctor_id = session['user_id']
    full_name = request.form.get('full_name')
    email = request.form.get('email')
    phone = request.form.get('phone')
    license_number = request.form.get('license_number')
    specialization = request.form.get('specialization')
    years_experience = request.form.get('years_experience')
    hospital_name = request.form.get('hospital_name', '')
    address = request.form.get('address', '')

    conn = get_db()
    c = conn.cursor()
    c.execute("""UPDATE doctors
                 SET full_name=?, email=?, phone=?, license_number=?, specialization=?,
                     years_experience=?, hospital_name=?, address=?
                 WHERE id=?""",
              (full_name, email, phone, license_number, specialization,
               years_experience, hospital_name, address, doctor_id))
    conn.commit()
    conn.close()

    flash("Profile updated successfully!", "success")
    return redirect(url_for('doctor_settings'))


@app.route('/doctor/settings/password', methods=['POST'])
def change_password():
    if session.get('user_type') != 'doctor':
        return redirect(url_for('doctor_login'))

    doctor_id = session['user_id']
    current_password = request.form.get('current_password', '')
    new_password = request.form.get('new_password', '')
    confirm_password = request.form.get('confirm_password', '')

    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT password FROM doctors WHERE id = ?", (doctor_id,))
    row = c.fetchone()

    if not row or not check_password_hash(row[0], current_password):
        flash("Current password incorrect", "danger")
        conn.close()
        return redirect(url_for('doctor_settings'))

    if new_password != confirm_password:
        flash("New passwords do not match", "danger")
        conn.close()
        return redirect(url_for('doctor_settings'))

    c.execute("UPDATE doctors SET password=? WHERE id=?", (generate_password_hash(new_password), doctor_id))
    conn.commit()
    conn.close()

    flash("Password updated successfully!", "success")
    return redirect(url_for('doctor_settings'))


@app.route('/doctor/settings/delete', methods=['POST'])
def doctor_settings_delete():
    if session.get('user_type') != 'doctor':
        return redirect(url_for('doctor_login'))

    doctor_id = session['user_id']
    conn = get_db()
    c = conn.cursor()
    c.execute("DELETE FROM doctors WHERE id = ?", (doctor_id,))
    conn.commit()
    conn.close()

    session.clear()
    flash("Your account has been deleted permanently.", "success")
    return redirect(url_for('home'))


# ── Doctor-patient assignment ──────────────────────────────────────────────────

@app.route('/assign_patient', methods=['POST'])
def assign_patient():
    if session.get('user_type') != 'doctor':
        return redirect(url_for('doctor_login'))
    doctor_id = session['user_id']
    patient_id = request.form.get('patient_id')
    conn = get_db()
    c = conn.cursor()
    try:
        c.execute("INSERT INTO doctor_patients (doctor_id, patient_id) VALUES (?, ?)",
                  (doctor_id, patient_id))
        conn.commit()
        flash("Patient assigned", "success")
    except sqlite3.IntegrityError:
        flash("Patient already assigned", "warning")
    finally:
        conn.close()
    return redirect(url_for('doctor_dashboard'))


@app.route('/unassign_patient', methods=['POST'])
def unassign_patient():
    if session.get('user_type') != 'doctor':
        return redirect(url_for('doctor_login'))
    doctor_id = session['user_id']
    patient_id = request.form.get('patient_id')
    conn = get_db()
    c = conn.cursor()
    c.execute("DELETE FROM doctor_patients WHERE doctor_id = ? AND patient_id = ?",
              (doctor_id, patient_id))
    conn.commit()
    conn.close()
    flash("Patient unassigned", "info")
    return redirect(url_for('doctor_dashboard'))


# ── Prediction ─────────────────────────────────────────────────────────────────

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if session.get('user_type') != 'patient':
        return redirect(url_for('patient_login'))

    if request.method == 'GET':
        conn = get_db()
        c = conn.cursor()
        c.execute("SELECT gender, date_of_birth FROM patients WHERE id = ?", (session['user_id'],))
        patient = c.fetchone()
        conn.close()

        auto_gender = 1 if str(patient["gender"]).lower() == "male" else 0
        auto_age = calculate_age(patient["date_of_birth"])
        extracted = session.get("extracted", {})
        extracted_preview = session.get("extracted_preview", {})

        return render_template("predict.html",
                               auto_gender=auto_gender,
                               auto_age=auto_age,
                               extracted=extracted,
                               extracted_preview=extracted_preview)

    required_fields = ["gender", "age", "bmi", "systolic_bp", "glucose", "hba1c", "triglycerides"]
    missing = [f for f in required_fields if not request.form.get(f)]
    if missing:
        flash("Please fill all medical values before prediction.", "warning")
        return redirect(url_for("predict"))

    try:
        input_data = {
            "gender": float(request.form["gender"]),
            "age": float(request.form["age"]),
            "bmi": float(request.form["bmi"]),
            "systolic_bp": float(request.form["systolic_bp"]),
            "glucose": float(request.form["glucose"]),
            "hba1c": float(request.form["hba1c"]),
            "triglycerides": float(request.form["triglycerides"])
        }
    except ValueError:
        flash("Invalid numeric values entered.", "danger")
        return redirect(url_for("predict"))

    # Training: scaler.fit_transform(X_train) where X_train was a named DataFrame
    #           model.fit(X_train_scaled, y_train) where X_train_scaled was a numpy array
    # So: scaler needs a named DataFrame, model needs a numpy array.
    FEATURE_COLUMNS = ["gender", "age", "bmi", "systolic_bp", "glucose", "hba1c", "triglycerides"]

    try:
        # Step 1: named DataFrame → scaler (scaler was fitted on a named DataFrame)
        X_df = pd.DataFrame([[input_data[col] for col in FEATURE_COLUMNS]],
                            columns=FEATURE_COLUMNS)
        # Step 2: scale → produces numpy array (no feature names)
        X_scaled_np = scaler.transform(X_df)
        # Step 3: pass numpy array to model (model was fitted on scaled numpy array)
        probs = model.predict_proba(X_scaled_np)[0]
        pred_class = int(model.predict(X_scaled_np)[0])
        class_map = {0: "Normal", 1: "Prediabetes", 2: "Diabetes"}
        final_status = class_map[pred_class]
        prob_normal = round(probs[0] * 100, 2)
        prob_prediabetes = round(probs[1] * 100, 2)
        prob_diabetes = round(probs[2] * 100, 2)
        diabetes_risk = round((probs[1] + probs[2]) * 100, 2)
    except Exception as e:
        print(f"❌ Prediction Error: {e}")
        flash("Prediction failed.", "danger")
        return redirect(url_for("predict"))

    try:
        report = generate_diabetes_report(input_data, final_status, probs[2])
    except Exception as e:
        print(f"❌ Report Generation Error: {e}")
        report = "AI report unavailable."

    conn = get_db()
    c = conn.cursor()
    c.execute("""INSERT INTO predictions
                 (patient_id, glucose, bmi, age, systolic_bp, hba1c, triglycerides,
                  prediction_result, risk_probability, recommendations)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
              (session['user_id'],
               input_data["glucose"], input_data["bmi"], input_data["age"],
               input_data["systolic_bp"], input_data["hba1c"], input_data["triglycerides"],
               final_status, diabetes_risk, report))
    conn.commit()
    record_id = c.lastrowid
    conn.close()

    session.pop("extracted", None)
    session.pop("extracted_preview", None)

    # Recommended doctors
    if final_status == "Normal":
        keywords = ["general", "physician", "family"]
    elif final_status == "Prediabetes":
        keywords = ["diabet", "metabolic"]
    else:
        keywords = ["endocr", "diabet"]

    conn = get_db()
    c = conn.cursor()
    query = " OR ".join(["LOWER(specialization) LIKE ?"] * len(keywords))
    params = [f"%{k}%" for k in keywords]
    c.execute(f"SELECT * FROM doctors WHERE {query}", params)
    recommended_doctors = c.fetchall()
    conn.close()

    return render_template(
        "result.html",
        prediction={
            "id": record_id,
            "prediction_result": final_status,
            "risk_probability": diabetes_risk,
            "prob_normal": prob_normal,
            "prob_prediabetes": prob_prediabetes,
            "prob_diabetes": prob_diabetes,
            "report": report,
            "glucose": input_data["glucose"],
            "hba1c": input_data["hba1c"],
            "bmi": input_data["bmi"],
            "age": input_data["age"],
            "systolic_bp": input_data["systolic_bp"],
            "triglycerides": input_data["triglycerides"],
            "gender": input_data["gender"]
        },
        recommended_doctors=recommended_doctors
    )


@app.route('/recommend_doctors')
def recommend_doctors_page():
    if session.get('user_type') != 'patient':
        return redirect(url_for('patient_login'))

    prediction_result = request.args.get("prediction_result", "").lower()

    if "normal" in prediction_result:
        keywords = ["general", "physician", "family"]
    elif "prediabetes" in prediction_result or "pre-diabetes" in prediction_result:
        keywords = ["diabet", "diabetologist", "metabolic"]
    elif "diabetes" in prediction_result:
        keywords = ["endocr", "diabet", "diabetes"]
    else:
        keywords = ["general"]

    conn = get_db()
    c = conn.cursor()
    where_clause = " OR ".join(["LOWER(specialization) LIKE ?"] * len(keywords))
    params = [f"%{k}%" for k in keywords]
    c.execute(f"SELECT * FROM doctors WHERE {where_clause}", params)
    recommended = c.fetchall()
    conn.close()

    return render_template("recommend_doctors.html", recommended_doctors=recommended)


@app.route('/choose_doctor', methods=['POST'])
def choose_doctor():
    if session.get('user_type') != 'patient':
        return redirect(url_for('patient_login'))

    patient_id = session['user_id']
    doctor_id = request.form.get("doctor_id")

    conn = get_db()
    c = conn.cursor()
    c.execute("""INSERT INTO doctor_patients (doctor_id, patient_id)
                 SELECT ?, ?
                 WHERE NOT EXISTS (
                     SELECT 1 FROM doctor_patients
                     WHERE doctor_id = ? AND patient_id = ?
                 )""", (doctor_id, patient_id, doctor_id, patient_id))
    conn.commit()
    conn.close()

    flash("Doctor assigned successfully!", "success")
    return redirect(url_for('dashboard'))


def auto_assign_doctor(patient_id, prediction_result):
    conn = get_db()
    c = conn.cursor()
    condition = prediction_result.lower()
    if "diabetes" in condition:
        specialization = "Diabetologist"
    elif "heart" in condition or "cardio" in condition:
        specialization = "Cardiologist"
    elif "kidney" in condition:
        specialization = "Nephrologist"
    else:
        specialization = "General"

    c.execute("""SELECT * FROM doctors WHERE specialization LIKE ?
                 ORDER BY years_experience DESC LIMIT 1""", (f'%{specialization}%',))
    doctor = c.fetchone()

    if doctor:
        c.execute("""INSERT INTO doctor_patients (doctor_id, patient_id)
                     SELECT ?, ? WHERE NOT EXISTS (
                         SELECT 1 FROM doctor_patients WHERE doctor_id = ? AND patient_id = ?
                     )""", (doctor["id"], patient_id, doctor["id"], patient_id))
    conn.commit()
    conn.close()


# ── Lab upload ─────────────────────────────────────────────────────────────────

@app.route('/upload_lab', methods=['GET', 'POST'])
def upload_lab():
    if session.get('user_type') != 'patient':
        return redirect(url_for('patient_login'))

    if request.method == 'GET':
        return render_template('upload_lab.html')

    try:
        report_type = request.form.get('report_type', '')
        notes = request.form.get('notes', '')
        file = request.files.get('report_file')

        if not file or file.filename == '':
            flash("No file selected", "danger")
            return redirect(url_for('upload_lab'))

        if not allowed_file(file.filename):
            flash("File type not allowed", "danger")
            return redirect(url_for('upload_lab'))

        timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')
        filename = secure_filename(f"{session['user_id']}_{timestamp}_{file.filename}")
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)

        conn = get_db()
        c = conn.cursor()
        c.execute("""INSERT INTO lab_reports (patient_id, report_type, file_name, file_path, notes, uploaded_at)
                     VALUES (?, ?, ?, ?, ?, datetime('now'))""",
                  (session['user_id'], report_type, filename, save_path, notes))
        conn.commit()
        conn.close()

        flash("Lab report uploaded successfully.", "success")
        return redirect(url_for('dashboard'))

    except Exception as e:
        print(f"\n===== UPLOAD LAB ERROR =====\n{e}\n============================\n")
        flash("Error: " + str(e), "danger")
        return redirect(url_for('upload_lab'))


@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    if 'user_id' not in session:
        abort(403)
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=False)


# ── History ────────────────────────────────────────────────────────────────────

@app.route('/history')
def history():
    if session.get('user_type') != 'patient':
        return redirect(url_for('patient_login'))

    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM patients WHERE id = ?", (session['user_id'],))
    patient = c.fetchone()
    age = calculate_age(patient["date_of_birth"])

    c.execute("""SELECT * FROM predictions WHERE patient_id = ?
                 ORDER BY created_at DESC""", (session['user_id'],))
    rows = c.fetchall()
    conn.close()

    predictions = []
    for r in rows:
        r = dict(r)
        if r.get("recommendations"):
            clean_html = bleach.clean(r["recommendations"], tags=ALLOWED_TAGS, strip=True)
            r["recommendations"] = Markup(clean_html)
        predictions.append(r)

    return render_template("history.html", predictions=predictions, patient=patient, age=age)


# ── Download text report ───────────────────────────────────────────────────────

@app.route('/download/<int:id>/pdf')
def download_pdf(id):
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM predictions WHERE id = ?", (id,))
    record = c.fetchone()
    conn.close()

    if not record:
        abort(404)

    # Patient info
    patient_id = record["patient_id"]

    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM patients WHERE id = ?", (patient_id,))
    patient = c.fetchone()
    conn.close()

    # Prepare report data
    ai_report = record["recommendations"]

    # ------------ PDF SETUP ------------
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)

    styles = getSampleStyleSheet()
    title = ParagraphStyle('title', parent=styles['Title'], fontSize=22, textColor=colors.HexColor('#3740ff'))
    header = ParagraphStyle('header', parent=styles['Heading2'], fontSize=16, textColor=colors.HexColor('#3740ff'))
    normal = styles['BodyText']

    story = []

    # ------------ HOSPITAL HEADER (NO LOGO NEEDED) ------------
    story.append(Paragraph("<b>DiaCare Hospital - Diabetes Report</b>", title))
    story.append(Spacer(1, 6))
    story.append(Paragraph("<i>Advanced AI-Assisted Health Evaluation</i>", normal))
    story.append(Spacer(1, 15))

    story.append(Paragraph("<b>Patient Information</b>", header))
    story.append(Spacer(1, 10))

    patient_table = [
        ["Name:", patient["full_name"]],
        ["Email:", patient["email"]],
        ["Age:", record["age"]],
        ["Date:", record["created_at"]],
    ]

    table = Table(patient_table, colWidths=[120, 300])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.whitesmoke),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(table)
    story.append(Spacer(1, 20))

    # ------------ PREDICTION SUMMARY ------------
    story.append(Paragraph("<b>Prediction Summary</b>", header))
    story.append(Spacer(1, 12))

    summary = f"""
    <b>Prediction Result:</b> {record['prediction_result']}<br/>
    <b>Risk Probability:</b> {record['risk_probability']}%<br/>
    """
    story.append(Paragraph(summary, normal))
    story.append(Spacer(1, 20))

    # ------------ HEALTH PARAMETERS ------------
    story.append(Paragraph("<b>Health Parameters</b>", header))
    story.append(Spacer(1, 12))

    param_table = [
        ["Glucose", record["glucose"]],
        ["Blood Pressure", record["blood_pressure"]],
        ["Skin Thickness", record["skin_thickness"]],
        ["Insulin", record["insulin"]],
        ["BMI", record["bmi"]],
        ["Pedigree Function", record["diabetes_pedigree"]],
    ]

    table2 = Table(param_table, colWidths=[180, 150])
    table2.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f0f2ff')),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
    ]))

    story.append(table2)
    story.append(Spacer(1, 25))

    # ------------ AI GENERATED REPORT ------------
    story.append(Paragraph("<b>AI-Generated Medical Interpretation</b>", header))
    story.append(Spacer(1, 10))

    story.append(Paragraph(ai_report.replace("\n", "<br/>"), normal))
    story.append(Spacer(1, 30))

    # ------------ DISCLAIMER ------------
    disclaimer = """
    <b>Disclaimer:</b><br/>
    This report is generated using AI-based analysis of the provided health parameters.  
    It is not a medical diagnosis. Please consult a certified healthcare professional for  
    clinical evaluation and treatment.
    """
    story.append(Paragraph(disclaimer, ParagraphStyle("disc", parent=normal, textColor=colors.red)))
    story.append(Spacer(1, 15))

    doc.build(story)
    buffer.seek(0)

    return send_file(buffer, as_attachment=True, download_name="DiaCare_Report.pdf", mimetype='application/pdf')


# ── Ask AI (SSE streaming) ─────────────────────────────────────────────────────

@app.route("/ask", methods=["GET"])
def ask_ai():
    if session.get("user_type") != "patient":
        return redirect(url_for("patient_login"))

    question = request.args.get("question", "").strip()

    if not question:
        return render_template("ask.html")

    blocked = ["gandhi", "politics", "history", "actor", "movie", "president"]
    if any(w in question.lower() for w in blocked):
        def block():
            yield 'data: {"type":"chunk","content":"\\u26a0\\ufe0f I can only answer diabetes and health-related questions."}\n\n'
            yield 'data: {"type":"done"}\n\n'
        return Response(block(), mimetype="text/event-stream")

    if not deepseek:
        def fail():
            yield 'data: {"type":"chunk","content":"AI service is currently unavailable."}\n\n'
            yield 'data: {"type":"done"}\n\n'
        return Response(fail(), mimetype="text/event-stream")

    def stream():
        try:
            response = deepseek.chat.completions.create(
                model="deepseek/deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are DiaCare AI, a medical diabetes assistant."},
                    {"role": "user", "content": question}
                ],
                temperature=0.3,
                max_tokens=600,
                stream=True
            )
            for chunk in response:
                delta = getattr(chunk.choices[0], "delta", None)
                if delta and getattr(delta, "content", None):
                    yield f'data: {json.dumps({"type": "chunk", "content": delta.content})}\n\n'
            yield 'data: {"type":"done"}\n\n'
        except Exception as e:
            yield f'data: {json.dumps({"type": "chunk", "content": f"AI error: {str(e)}"})}\n\n'
            yield 'data: {"type":"done"}\n\n'

    return Response(stream(), mimetype="text/event-stream")


@app.route('/ask_simple', methods=['POST'])
def ask_ai_simple():
    if session.get('user_type') != 'patient':
        return jsonify({"error": "Please login first"}), 403

    data = request.get_json()
    query = data.get('question', '').strip() if data else ''

    if not query:
        return jsonify({"error": "Empty question"}), 400

    if not deepseek:
        return jsonify({"error": "AI service unavailable"}), 503

    try:
        context_text = "Using general medical knowledge."
        if retrieve_sections:
            try:
                sections = retrieve_sections(query, k=3)
                if sections:
                    context_text = "\n\n".join(sections)
            except Exception:
                pass

        system_prompt = "You are DiaCare AI, a helpful diabetes education assistant. Provide clear, concise answers to patient questions."
        user_prompt = f"Patient Report Context:\n{context_text}\n\nUser Question:\n{query}"

        response = deepseek.chat.completions.create(
            model="deepseek/deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=600
        )
        answer = response.choices[0].message.content
        return jsonify({"success": True, "answer": answer})
    except Exception as e:
        logger.error(f"AI error: {e}")
        return jsonify({"error": "Failed to generate response"}), 500


# ── Lab extraction routes ──────────────────────────────────────────────────────

@app.route("/extract_lab", methods=["POST"])
def extract_lab():
    if session.get("user_type") != "patient":
        flash("Please login as a patient", "danger")
        return redirect(url_for("patient_login"))

    file = request.files.get("lab_file")
    if not file or file.filename == '':
        flash("No lab file uploaded", "danger")
        return redirect(url_for("predict"))

    if not file.filename.lower().endswith('.pdf'):
        flash("Please upload PDF files only", "danger")
        return redirect(url_for("predict"))

    try:
        text = extract_text_from_file(file)
        if not text:
            flash("Could not read the PDF. Please try another file.", "danger")
            return redirect(url_for("predict"))

        data = extract_medical_values_llm(text)
        if not data:
            flash("No values found in the report. Please enter manually.", "warning")
            return redirect(url_for("predict"))

        existing_combined = merge_extracted_data(
            session.get("extracted", {}), session.get("extracted_preview", {}))

        extracted_values = {k: data.get(k) for k in
                            ["glucose", "hba1c", "triglycerides", "systolic_bp", "bmi"]}
        merged = merge_extracted_data(existing_combined, extracted_values)
        session["extracted_preview"] = merged

        count = len([v for v in extracted_values.values() if v is not None])
        if count == 0:
            flash("No medical values found in this report.", "warning")
        else:
            flash(f"✅ Extracted {count} values from report. Click 'Apply' to use them.", "success")

    except Exception as e:
        import traceback; traceback.print_exc()
        flash(f"Error processing report: {str(e)}", "danger")

    return redirect(url_for("predict"))


@app.route("/extract_vitals", methods=["POST"])
def extract_vitals():
    if session.get("user_type") != "patient":
        flash("Please login as a patient", "danger")
        return redirect(url_for("patient_login"))

    file = request.files.get("vitals_file")
    if not file or file.filename == '':
        flash("No vitals file uploaded", "danger")
        return redirect(url_for("predict"))

    if not file.filename.lower().endswith('.pdf'):
        flash("Please upload PDF files only", "danger")
        return redirect(url_for("predict"))

    try:
        text = extract_text_from_file(file)
        if not text:
            flash("Could not read the PDF. Please try another file.", "danger")
            return redirect(url_for("predict"))

        data = extract_medical_values_llm(text)
        if not data:
            flash("No values found in the report. Please enter manually.", "warning")
            return redirect(url_for("predict"))

        existing_combined = merge_extracted_data(
            session.get("extracted", {}), session.get("extracted_preview", {}))

        extracted_values = {k: data.get(k) for k in
                            ["glucose", "hba1c", "triglycerides", "systolic_bp", "bmi"]}
        merged = merge_extracted_data(existing_combined, extracted_values)
        session["extracted_preview"] = merged

        count = len([v for v in extracted_values.values() if v is not None])
        if count == 0:
            flash("No medical values found in this report.", "warning")
        else:
            flash(f"✅ Extracted {count} values from report. Click 'Apply' to use them.", "success")

    except Exception as e:
        import traceback; traceback.print_exc()
        flash(f"Error processing report: {str(e)}", "danger")

    return redirect(url_for("predict"))


@app.route("/apply_extracted", methods=["POST"])
def apply_extracted():
    if session.get("user_type") != "patient":
        return redirect(url_for("patient_login"))

    preview_data = session.get("extracted_preview", {})
    if not preview_data:
        flash("No extracted values to apply.", "warning")
        return redirect(url_for("predict"))

    merged_applied = merge_extracted_data(session.get("extracted", {}), preview_data)
    session["extracted"] = merged_applied
    session.pop("extracted_preview", None)

    applied_count = len([v for v in merged_applied.values() if v is not None])
    flash(f"✅ Applied {applied_count} values to the form. You can upload more reports or submit prediction.",
          "predict")
    return redirect(url_for("predict"))


# ── Appointments ───────────────────────────────────────────────────────────────

@app.route("/patient/appointments/new", methods=["GET", "POST"])
def new_appointment():
    if session.get("user_type") != "patient":
        return redirect(url_for("patient_login"))

    doctor_id = request.args.get("doctor_id")

    if request.method == "POST":
        doctor_id = request.form.get("doctor_id")
        appt_date = request.form.get("date")
        appt_time = request.form.get("time")
        reason = request.form.get("reason")

        if not all([doctor_id, appt_date, appt_time, reason]):
            flash("All fields are required.", "danger")
            return redirect(request.url)

        conn = get_db()
        c = conn.cursor()
        c.execute("""INSERT INTO appointments
                     (patient_id, doctor_id, appointment_date, appointment_time, reason)
                     VALUES (?, ?, ?, ?, ?)""",
                  (session["user_id"], doctor_id, appt_date, appt_time, reason))
        conn.commit()
        conn.close()

        flash("Appointment request sent successfully.", "success")
        return redirect(url_for("dashboard"))

    return render_template("take_appointment.html", doctor_id=doctor_id)


@app.route("/patient/appointments")
def patient_appointments():
    if session.get("user_type") != "patient":
        return redirect(url_for("patient_login"))

    conn = get_db()
    c = conn.cursor()
    c.execute("""SELECT a.id, a.appointment_date, a.appointment_time, a.reason,
                        a.status, a.created_at, d.full_name AS doctor_name, d.specialization
                 FROM appointments a
                 JOIN doctors d ON d.id = a.doctor_id
                 WHERE a.patient_id = ?
                 ORDER BY a.created_at DESC""", (session["user_id"],))
    appointments = c.fetchall()
    conn.close()
    return render_template("patient_appointments.html", appointments=appointments)


@app.route("/patient/appointments/cancel", methods=["POST"])
def cancel_appointment():
    if session.get("user_type") != "patient":
        return redirect(url_for("patient_login"))

    appointment_id = request.form.get("appointment_id")
    if not appointment_id:
        flash("Invalid appointment.", "danger")
        return redirect(url_for("patient_appointments"))

    conn = get_db()
    c = conn.cursor()
    c.execute("""UPDATE appointments SET status = 'Cancelled'
                 WHERE id = ? AND patient_id = ? AND status = 'Pending'""",
              (appointment_id, session["user_id"]))
    if c.rowcount == 0:
        flash("Only pending appointments can be cancelled.", "warning")
    else:
        flash("Appointment cancelled successfully.", "info")
    conn.commit()
    conn.close()
    return redirect(url_for("patient_appointments"))


@app.route("/doctor/appointments")
def doctor_appointments():
    if session.get("user_type") != "doctor":
        return redirect(url_for("doctor_login"))

    conn = get_db()
    c = conn.cursor()
    c.execute("""SELECT a.id, a.appointment_date, a.appointment_time, a.reason,
                        a.status, a.created_at, p.full_name AS patient_name, p.phone AS patient_phone
                 FROM appointments a
                 JOIN patients p ON p.id = a.patient_id
                 WHERE a.doctor_id = ?
                 ORDER BY a.created_at DESC""", (session["user_id"],))
    appointments = c.fetchall()
    conn.close()
    return render_template("doctor_appointments.html", appointments=appointments)


@app.route("/doctor/appointments/update", methods=["POST"])
def update_appointment():
    if session.get("user_type") != "doctor":
        return redirect(url_for("doctor_login"))

    appt_id = request.form.get("appt_id")
    status = request.form.get("status")

    if status not in ["Accepted", "Rejected"]:
        flash("Invalid action.", "danger")
        return redirect(url_for("doctor_dashboard"))

    conn = get_db()
    c = conn.cursor()
    c.execute("UPDATE appointments SET status = ? WHERE id = ? AND doctor_id = ?",
              (status, appt_id, session["user_id"]))
    conn.commit()
    conn.close()

    flash(f"Appointment {status.lower()} successfully.", "success")
    return redirect(url_for("doctor_dashboard"))


# ============= Run =============
if __name__ == "__main__":
    print("\n🚀 DiaCare Flask server starting")
    print("🌐 Open in browser → http://127.0.0.1:5004\n")

    logger.info("Starting DiaCare app...")
    logger.info("RAG: %s", "Active" if retrieve_sections else "Unavailable")
    logger.info("DeepSeek: %s", "Configured" if deepseek else "Not configured")

    logging.getLogger("werkzeug").setLevel(logging.INFO)

    app.run(
        host="127.0.0.1",
        port=5004,
        debug=True,
        use_reloader=False
    )
