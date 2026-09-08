import datetime
import io
import json
import math
from typing import List, Optional, Tuple

import bcrypt
from google import genai
from google.genai import types
from PIL import Image
import pypdf
import streamlit as st
from supabase import Client, create_client

# ==========================================
# 1. INITIALIZATION & DATABASE
# ==========================================

st.set_page_config(
    page_title="Multi-User AI Flashcard Hub", page_icon="🧠", layout="centered"
)


@st.cache_resource
def init_supabase() -> Client:
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)


supabase = init_supabase()

# Initialize Google GenAI Client
try:
    api_key_val = st.secrets.get("GEMINI_API_KEY")
    if not api_key_val or not str(api_key_val).strip():
        raise ValueError("GEMINI_API_KEY is empty or missing from secrets.")
    
    client = genai.Client(api_key=api_key_val)
except Exception as e:
    st.error(f"Missing or invalid `GEMINI_API_KEY` in Streamlit secrets: {e}")
    st.stop()
# GEMINI MODEL IDENTIFIER
MODEL_NAME = "gemini-3.5-flash"

if "user" not in st.session_state:
    st.session_state.user = None

# ==========================================
# 2. AUTHENTICATION HELPERS
# ==========================================


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode(
        "utf-8"
    )


def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))


def register_user(username: str, password: str) -> Tuple[bool, str]:
    hashed = hash_password(password)
    res = (
        supabase.table("users")
        .select("id")
        .eq("username", username)
        .execute()
    )
    if res.data:
        return False, "Username already exists."

    supabase.table("users").insert(
        {"username": username, "password_hash": hashed}
    ).execute()
    return True, "Account created successfully! Please log in."


def login_user(username: str, password: str) -> Tuple[bool, str]:
    res = (
        supabase.table("users")
        .select("*")
        .eq("username", username)
        .execute()
    )
    if not res.data:
        return False, "User not found."

    user_data = res.data[0]
    if verify_password(password, user_data["password_hash"]):
        st.session_state.user = {
            "id": user_data["id"],
            "username": user_data["username"],
        }
        return True, "Logged in!"
    return False, "Invalid password."


# ==========================================
# 3. DATABASE OPERATIONS
# ==========================================


def save_cards(user_id: str, cards: List[dict]):
    today_str = datetime.date.today().isoformat()
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    records = []
    for card in cards:
        records.append(
            {
                "user_id": user_id,
                "subject": card.get("subject", "General").strip(),
                "question": card.get("question", ""),
                "diagram": card.get("diagram", None),
                "answer": card.get("answer", ""),
                "last_reviewed": now_str,
                "interval": 0,
                "repetition": 0,
                "efactor": 2.5,
                "next_review": today_str,
            }
        )
    if records:
        supabase.table("flashcards").insert(records).execute()


def get_cards_by_subject(user_id: str, subject: str = "All") -> List[dict]:
    query = supabase.table("flashcards").select("*").eq("user_id", user_id)
    if subject != "All":
        query = query.eq("subject", subject)
    res = query.execute()
    return res.data or []


def get_all_subjects(user_id: str) -> List[str]:
    res = (
        supabase.table("flashcards")
        .select("subject")
        .eq("user_id", user_id)
        .execute()
    )
    if not res.data:
        return []
    return sorted(list({r["subject"] for r in res.data if r.get("subject")}))


def update_card_review(
    card_id: str, quality: int, current_rep: int, current_int: int, current_ef: float
):
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

    new_ef = current_ef + (
        0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02)
    )
    if new_ef < 1.3:
        new_ef = 1.3

    next_date = (
        datetime.date.today() + datetime.timedelta(days=new_int)
    ).isoformat()
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    supabase.table("flashcards").update(
        {
            "last_reviewed": now_str,
            "interval": new_int,
            "repetition": rep,
            "efactor": new_ef,
            "next_review": next_date,
        }
    ).eq("id", card_id).execute()


def delete_card(card_id: str):
    supabase.table("flashcards").delete().eq("id", card_id).execute()


# ==========================================
# 4. HELPER & GEMINI FUNCTIONS
# ==========================================


def extract_text_from_pdf(pdf_file) -> str:
    try:
        reader = pypdf.PdfReader(pdf_file)
        extracted_text = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                extracted_text.append(text)
        return "\n".join(extracted_text)
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return ""


def get_review_status(last_reviewed_str: Optional[str]) -> Tuple[str, bool]:
    if not last_reviewed_str:
        return "Needs Review", False
    try:
        last_date = datetime.datetime.strptime(
            last_reviewed_str, "%Y-%m-%d %H:%M:%S"
        ).date()
        days_passed = (datetime.date.today() - last_date).days
        return (
            ("Up to Date", True) if days_passed <= 3 else ("Due for Review", False)
        )
    except ValueError:
        return "Needs Review", False


def generate_cards_with_gemini(
    contents_input, num_cards: int = 5, existing_subjects: Optional[List[str]] = None
) -> dict:
    if not existing_subjects:
        existing_subjects = ["General"]
    existing_str = ", ".join(f'"{s}"' for s in existing_subjects)

    system_prompt = f"""
    Analyze the provided content and generate {num_cards} flashcards.

    CATEGORIZATION & FORMATTING RULES:
    1. Prefer choosing a subject from this existing list if it fits well: [{existing_str}].
    2. If NONE fit, create a NEW concise subject name (1-3 words).
    3. MATHEMATICS & FORMULAS: Format all equations using LaTeX ($x^2$ or $$\\int x$$).
    4. DIAGRAMS: Generate ASCII art in "diagram" if useful, otherwise null.
    """

    card_schema = types.Schema(
        type=types.Type.OBJECT,
        properties={
            "subject": types.Schema(type=types.Type.STRING),
            "question": types.Schema(type=types.Type.STRING),
            "diagram": types.Schema(type=types.Type.STRING, nullable=True),
            "answer": types.Schema(type=types.Type.STRING),
        },
        required=["subject", "question", "answer"],
    )

    response_schema = types.Schema(
        type=types.Type.OBJECT,
        properties={
            "cards": types.Schema(
                type=types.Type.ARRAY, items=card_schema
            )
        },
        required=["cards"],
    )

    contents = []
    if isinstance(contents_input, Image.Image):
        contents.append(contents_input)
        contents.append(f"Generate {num_cards} flashcards from this image.")
    else:
        contents.append(f"Content:\n{contents_input[:4000]}")

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
                response_schema=response_schema,
            ),
        )
        data = json.loads(response.text)
        return {"success": True, "cards": data.get("cards", [])}
    except Exception as e:
        return {"success": False, "error": str(e)}


def evaluate_answer(user_ans: str, correct_ans: str, question: str) -> dict:
    eval_prompt = f"""
    Question: {question}
    Correct Answer: {correct_ans}
    User Answer: {user_ans}
    """

    eval_schema = types.Schema(
        type=types.Type.OBJECT,
        properties={
            "is_correct": types.Schema(type=types.Type.BOOLEAN),
            "quality": types.Schema(type=types.Type.INTEGER),
            "feedback": types.Schema(type=types.Type.STRING),
        },
        required=["is_correct", "quality", "feedback"],
    )

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=eval_prompt,
            config=types.GenerateContentConfig(
                system_instruction="You evaluate flashcard quiz responses. Assess accuracy, rate quality between 0 and 5, and provide brief feedback.",
                response_mime_type="application/json",
                response_schema=eval_schema,
            ),
        )
        return json.loads(response.text)
    except Exception:
        return {
            "is_correct": False,
            "quality": 1,
            "feedback": "Evaluation failed.",
        }


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
tab1, tab2, tab3 = st.tabs(
    ["⚡ Generate Cards", "🎴 Smart Quiz", "📚 Dashboard"]
)

# TAB 1: GENERATE
with tab1:
    st.subheader("Generate New Flashcards")
    uploaded_file = st.file_uploader(
        "Upload (.txt, .pdf, .png, .jpg)", type=["txt", "pdf", "png", "jpg", "jpeg"]
    )
    text_input = st.text_area("Or paste notes:", height=150)
    card_count = st.number_input(
        "Card count", min_value=1, max_value=20, value=5
    )

    if st.button("Generate Flashcards", type="primary"):
        payload = None
        if uploaded_file:
            ext = uploaded_file.name.split(".")[-1].lower()
            if ext == "txt":
                payload = uploaded_file.read().decode("utf-8")
            elif ext == "pdf":
                payload = extract_text_from_pdf(uploaded_file)
            elif ext in ["png", "jpg", "jpeg"]:
                payload = Image.open(uploaded_file)
        elif text_input.strip():
            payload = text_input.strip()

        if payload:
            with st.spinner("Generating via Gemini AI..."):
                existing = get_all_subjects(user["id"])
                res = generate_cards_with_gemini(
                    payload, card_count, existing
                )
                if res["success"]:
                    save_cards(user["id"], res["cards"])
                    st.success(f"Generated {len(res['cards'])} cards!")
                    st.rerun()
                else:
                    st.error(res["error"])
        else:
            st.warning("Please upload a file or paste text content first.")

# TAB 2: QUIZ
with tab2:
    st.subheader("Smart Spaced-Repetition Quiz")
    subjects = get_all_subjects(user["id"])
    if not subjects:
        st.info("No cards found. Generate cards to get started!")
    else:
        sel_sub = st.selectbox("Select Subject", options=["All"] + subjects)

        if "quiz_cards" not in st.session_state or st.button(
            "Start Review Session"
        ):
            st.session_state.quiz_cards = get_cards_by_subject(
                user["id"], sel_sub
            )
            st.session_state.q_idx = 0
            st.session_state.show_ans = False
            st.session_state.eval = None

        cards = st.session_state.get("quiz_cards", [])
        if cards:
            idx = st.session_state.q_idx
            card = cards[idx]
            status_label, _ = get_review_status(card["last_reviewed"])

            st.caption(
                f"Card {idx + 1} of {len(cards)} | Subject: {card['subject']} | Status: {status_label}"
            )
            with st.container(border=True):
                st.markdown("### Q:")
                st.markdown(card["question"])
                if card.get("diagram"):
                    st.code(card["diagram"], language="text")

                user_ans = st.text_input("Your Answer:", key=f"ans_{idx}")
                if st.button("Submit Answer", type="primary", key=f"sub_{idx}"):
                    eval_res = evaluate_answer(
                        user_ans, card["answer"], card["question"]
                    )
                    st.session_state.eval = eval_res
                    st.session_state.show_ans = True
                    update_card_review(
                        card["id"],
                        eval_res.get("quality", 3),
                        card["repetition"],
                        card["interval"],
                        card["efactor"],
                    )

                if st.session_state.get("eval"):
                    res = st.session_state.eval
                    st.divider()
                    if res.get("is_correct"):
                        st.success(f"Feedback: {res.get('feedback', '')}")
                    else:
                        st.error(f"Feedback: {res.get('feedback', '')}")

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
    filter_sub = st.selectbox(
        "Filter Subject", options=["All"] + subjects, key="dash_sub"
    )
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
