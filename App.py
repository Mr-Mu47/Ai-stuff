# python -m streamlit run App.py
import json
import datetime
import math
import bcrypt
import streamlit as st
from google import genai
from google.genai import types, errors
from supabase import create_client, Client
from PIL import Image
import pypdf

# ==========================================
# 1. INITIALIZATION & DATABASE
# ==========================================

st.set_page_config(page_title="Multi-User AI Flashcard Hub", page_icon="🧠", layout="centered")

@st.cache_resource
def init_supabase() -> Client:
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

supabase = init_supabase()
client = genai.Client()

# Session State for Auth
if "user" not in st.session_state:
    st.session_state.user = None

# ==========================================
# 2. AUTHENTICATION FUNCTIONS
# ==========================================

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))

def register_user(username, password):
    hashed = hash_password(password)
    res = supabase.table("users").select("id").eq("username", username).execute()
    if res.data:
        return False, "Username already exists."
    
    supabase.table("users").insert({"username": username, "password_hash": hashed}).execute()
    return True, "Account created successfully! Please log in."

def login_user(username, password):
    res = supabase.table("users").select("*").eq("username", username).execute()
    if not res.data:
        return False, "User not found."
    
    user_data = res.data[0]
    if verify_password(password, user_data["password_hash"]):
        st.session_state.user = {"id": user_data["id"], "username": user_data["username"]}
        return True, "Logged in!"
    return False, "Invalid password."

# ==========================================
# 3. DATABASE OPERATIONS (USER SCOPED)
# ==========================================

def save_cards(user_id, subject, cards):
    today_str = datetime.date.today().isoformat()
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    records = []
    for card in cards:
        records.append({
            "user_id": user_id,
            "subject": subject,
            "question": card.get("question", ""),
            "diagram": card.get("diagram", None),
            "answer": card.get("answer", ""),
            "last_reviewed": now_str,
            "interval": 0,
            "repetition": 0,
            "efactor": 2.5,
            "next_review": today_str
        })
    supabase.table("flashcards").insert(records).execute()

def get_cards_by_subject(user_id, subject="All"):
    query = supabase.table("flashcards").select("*").eq("user_id", user_id)
    if subject != "All":
        query = query.eq("subject", subject)
    res = query.execute()
    return res.data or []

def get_all_subjects(user_id):
    res = supabase.table("flashcards").select("subject").eq("user_id", user_id).execute()
    if not res.data:
        return []
    return list(set([r["subject"] for r in res.data if r["subject"]]))

def update_card_review(card_id, quality, current_rep, current_int, current_ef):
    if quality >= 3:
        if current_rep == 0:
            new_int = 1
        elif current_rep == 1:
            new_int = 6
        else:
            new_int = math.ceil(current_int * current_ef)
        rep = current_rep + 1
    else:
        rep = 0
        new_int = 1

    new_ef = current_ef + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
    if new_ef < 1.3:
        new_ef = 1.3

    next_date = (datetime.date.today() + datetime.timedelta(days=new_int)).isoformat()
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    supabase.table("flashcards").update({
        "last_reviewed": now_str,
        "interval": new_int,
        "repetition": rep,
        "efactor": new_ef,
        "next_review": next_date
    }).eq("id", card_id).execute()

def delete_card(card_id):
    supabase.table("flashcards").delete().eq("id", card_id).execute()

# ==========================================
# 4. HELPER & GEMINI FUNCTIONS
# ==========================================

def extract_text_from_pdf(pdf_file):
    reader = pypdf.PdfReader(pdf_file)
    extracted_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            extracted_text += text + "\n"
    return extracted_text

def get_review_status(last_reviewed_str):
    if not last_reviewed_str:
        return "Needs Review", False
    try:
        last_date = datetime.datetime.strptime(last_reviewed_str, "%Y-%m-%d %H:%M:%S").date()
        days_passed = (datetime.date.today() - last_date).days
        return ("Up to Date", True) if days_passed <= 3 else ("Due for Review", False)
    except ValueError:
        return "Needs Review", False

def generate_cards_with_gemini(contents_input, num_cards=5, existing_subjects=None):
    if not existing_subjects:
        existing_subjects = ["General"]
    existing_str = ", ".join(f'"{s}"' for s in existing_subjects)

    prompt = f"""
    Analyze the provided content and generate {num_cards} flashcards.
    CATEGORIZATION & FORMATTING RULES:
    1. Prefer choosing a subject from this existing list if it fits well: [{existing_str}].
    2. If NONE fit, create a NEW concise subject name (1-3 words).
    3. MATHEMATICS & FORMULAS: Format all equations, formulas, and math symbols using LaTeX delimiters.
       - Use single dollars for inline math (e.g. $x^2 + y^2 = r^2$).
       - Use double dollars for display-style equations (e.g. $$\\int_{{a}}^{{b}} f(x) \\, dx$$).
    4. DIAGRAMS: If a question benefits from a visual diagram, generate a clean ASCII art diagram inside the "diagram" key. If not needed, set to null.
    """

    json_schema = types.Schema(
        type=types.Type.ARRAY,
        items=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "subject": types.Schema(type=types.Type.STRING),
                "question": types.Schema(type=types.Type.STRING),
                "diagram": types.Schema(type=types.Type.STRING, nullable=True),
                "answer": types.Schema(type=types.Type.STRING),
            },
            required=["subject", "question", "answer"]
        )
    )

    contents = [prompt] + contents_input if isinstance(contents_input, list) else f"{prompt}\n\nContent:\n{contents_input[:4000]}"

    try:
        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=json_schema,
            )
        )
        return {"success": True, "cards": json.loads(response.text.strip())}
    except Exception as e:
        return {"success": False, "error": str(e)}

def evaluate_answer(user_ans, correct_ans, question):
    eval_prompt = f"Question: {question}\nCorrect: {correct_ans}\nUser: {user_ans}\nRate quality (0-5) and provide short feedback."
    json_schema = types.Schema(
        type=types.Type.OBJECT,
        properties={
            "is_correct": types.Schema(type=types.Type.BOOLEAN),
            "quality": types.Schema(type=types.Type.INTEGER),
            "feedback": types.Schema(type=types.Type.STRING),
        },
        required=["is_correct", "quality", "feedback"]
    )
    try:
        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=eval_prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=json_schema,
            )
        )
        return json.loads(response.text.strip())
    except Exception:
        return {"is_correct": False, "quality": 1, "feedback": "Evaluation failed."}

# ==========================================
# 5. AUTHENTICATION UI
# ==========================================

if not st.session_state.user:
    st.title("🧠 AI Flashcard Hub")
    auth_tab1, auth_tab2 = st.tabs(["🔐 Login", "📝 Register"])

    with auth_tab1:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Login", type="primary"):
                success, msg = login_user(username, password)
                if success:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)

    with auth_tab2:
        with st.form("register_form"):
            new_user = st.text_input("New Username")
            new_pass = st.text_input("New Password", type="password")
            if st.form_submit_button("Create Account"):
                if new_user and new_pass:
                    success, msg = register_user(new_user, new_pass)
                    if success:
                        st.success(msg)
                    else:
                        st.error(msg)
                else:
                    st.warning("Please fill in all fields.")
    st.stop()

# ==========================================
# 6. LOGGED-IN APP UI
# ==========================================

user = st.session_state.user
st.sidebar.write(f"Logged in as: **{user['username']}**")
if st.sidebar.button("Logout"):
    st.session_state.user = None
    st.rerun()

st.title("🧠 AI Flashcard Hub")
tab1, tab2, tab3 = st.tabs(["⚡ Generate Cards", "🎴 Smart Quiz", "📚 Dashboard"])

# TAB 1: GENERATE
with tab1:
    st.subheader("Generate New Flashcards")
    uploaded_file = st.file_uploader("Upload (.txt, .pdf, .png, .jpg)", type=["txt", "pdf", "png", "jpg", "jpeg"])
    text_input = st.text_area("Or paste notes:", height=150)
    card_count = st.number_input("Card count", min_value=1, max_value=20, value=5)

    if st.button("Generate Flashcards", type="primary"):
        payload = None
        if uploaded_file:
            ext = uploaded_file.name.split(".")[-1].lower()
            if ext == "txt":
                payload = uploaded_file.read().decode("utf-8")
            elif ext == "pdf":
                payload = extract_text_from_pdf(uploaded_file)
            elif ext in ["png", "jpg", "jpeg"]:
                payload = [Image.open(uploaded_file)]
        elif text_input.strip():
            payload = text_input.strip()

        if payload:
            with st.spinner("Generating..."):
                existing = get_all_subjects(user["id"])
                res = generate_cards_with_gemini(payload, card_count, existing)
                if res["success"]:
                    for c in res["cards"]:
                        save_cards(user["id"], c.get("subject", "General").strip(), [c])
                    st.success(f"Generated {len(res['cards'])} cards!")
                    st.rerun()
                else:
                    st.error(res["error"])

# TAB 2: QUIZ
with tab2:
    st.subheader("Smart Spaced-Repetition Quiz")
    subjects = get_all_subjects(user["id"])
    if not subjects:
        st.info("No cards found. Generate cards to get started!")
    else:
        sel_sub = st.selectbox("Select Subject", options=["All"] + subjects)
        
        if "quiz_cards" not in st.session_state or st.button("Start Review Session"):
            st.session_state.quiz_cards = get_cards_by_subject(user["id"], sel_sub)
            st.session_state.q_idx = 0
            st.session_state.show_ans = False
            st.session_state.eval = None

        cards = st.session_state.get("quiz_cards", [])
        if cards:
            idx = st.session_state.q_idx
            card = cards[idx]
            status_label, _ = get_review_status(card["last_reviewed"])

            st.caption(f"Card {idx + 1} of {len(cards)} | Subject: {card['subject']} | Status: {status_label}")
            with st.container(border=True):
                st.markdown("### Q:")
                st.markdown(card["question"])
                if card.get("diagram"):
                    st.code(card["diagram"], language="text")

                user_ans = st.text_input("Your Answer:", key=f"ans_{idx}")
                if st.button("Submit Answer", type="primary", key=f"sub_{idx}"):
                    eval_res = evaluate_answer(user_ans, card["answer"], card["question"])
                    st.session_state.eval = eval_res
                    st.session_state.show_ans = True
                    update_card_review(card["id"], eval_res.get("quality", 3), card["repetition"], card["interval"], card["efactor"])

                if st.session_state.get("eval"):
                    res = st.session_state.eval
                    st.divider()
                    st.success(f"Feedback: {res['feedback']}") if res["is_correct"] else st.error(f"Feedback: {res['feedback']}")

                if st.session_state.get("show_ans"):
                    st.markdown("**Expected Answer:**")
                    st.markdown(card["answer"])

            c1, c2, c3 = st.columns([1, 2, 1])
            with c1:
                if st.button("⬅️ Previous", disabled=(idx == 0)):
                    st.session_state.q_idx -= 1
                    st.session_state.show_ans = False
                    st.session_state.eval = None
                    st.rerun()
            with c3:
                if st.button("Next ➡️", disabled=(idx >= len(cards) - 1)):
                    st.session_state.q_idx += 1
                    st.session_state.show_ans = False
                    st.session_state.eval = None
                    st.rerun()

# TAB 3: DASHBOARD
with tab3:
    st.subheader("📚 Saved Flashcards Dashboard")
    subjects = get_all_subjects(user["id"])
    filter_sub = st.selectbox("Filter Subject", options=["All"] + subjects, key="dash_sub")
    all_data = get_cards_by_subject(user["id"], filter_sub)

    if not all_data:
        st.info("No saved cards.")
    else:
        for c in all_data:
            with st.container(border=True):
                r1, r2, r3, r4 = st.columns([1.5, 3, 3, 0.8])
                r1.markdown(f"**{c['subject']}**")
                r2.markdown(c["question"])
                r3.markdown(c["answer"])
                if r4.button("🗑️", key=f"del_{c['id']}"):
                    delete_card(c["id"])
                    st.rerun()