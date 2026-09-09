import os
import math
import datetime
import bcrypt
import json
import streamlit as st
import pandas as pd
from supabase import create_client, Client
from google import genai
from google.genai import types

# ==========================================
# 1. INITIALIZATION & CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Multi-User AI Flashcard Hub",
    page_icon="🧠",
    layout="wide"
)

# Initialize Supabase
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("Missing Supabase configuration. Set `SUPABASE_URL` and `SUPABASE_KEY` environment variables.")
    st.stop()

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Initialize Session State
if "user" not in st.session_state:
    st.session_state.user = None
if "model_cooldowns" not in st.session_state:
    st.session_state.model_cooldowns = {}

MODEL_CASCADE = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite"
]
COOLDOWN_PERIOD_SECONDS = 60

# ==========================================
# 2. GEMINI FALLBACK PIPELINE
# ==========================================

def get_gemini_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        st.error("Missing GEMINI_API_KEY environment variable.")
        st.stop()
    return genai.Client(api_key=api_key)

def call_gemini_with_fallback(prompt: str, response_schema=None, system_instruction: str = None) -> str:
    client = get_gemini_client()
    now = datetime.datetime.now(datetime.timezone.utc)
    
    config = types.GenerateContentConfig()
    if response_schema:
        config.response_mime_type = "application/json"
        config.response_schema = response_schema
    if system_instruction:
        config.system_instruction = system_instruction

    for model in MODEL_CASCADE:
        cooldown_until = st.session_state.model_cooldowns.get(model)
        if cooldown_until and now < cooldown_until:
            continue

        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=config
            )
            return response.text
        except Exception as e:
            error_msg = str(e).lower()
            if "429" in error_msg or "503" in error_msg or "quota" in error_msg:
                st.session_state.model_cooldowns[model] = now + datetime.timedelta(seconds=COOLDOWN_PERIOD_SECONDS)
                st.warning(f"Model {model} busy/rate-limited. Falling back to next available model...")
                continue
            else:
                st.error(f"Error calling {model}: {e}")
                raise e

    raise Exception("All Gemini models in the fallback pipeline are currently unavailable or rate-limited.")

# ==========================================
# 3. FORGETTING CURVE & SM-2 LOGIC
# ==========================================

def calculate_forgetting_curve(last_reviewed: str, interval: int) -> tuple[float, str, str]:
    """
    Calculates retention percentage (R) using Ebbinghaus Forgetting Curve: R = e^(-t / S)
    where t is elapsed time in days, and S is memory stability (interval in days).
    """
    if not last_reviewed:
        return 0.0, "🔴 High Memory Decay (Needs Review)", "#FF4B4B"

    now = datetime.datetime.now(datetime.timezone.utc)
    
    # Safely parse last_reviewed ISO string and enforce UTC timezone awareness
    try:
        clean_iso = str(last_reviewed).replace("Z", "+00:00")
        last_dt = datetime.datetime.fromisoformat(clean_iso)
        # Convert naive datetime to UTC aware if timezone info is missing
        if last_dt.tzinfo is None:
            last_dt = last_dt.replace(tzinfo=datetime.timezone.utc)
        else:
            last_dt = last_dt.astimezone(datetime.timezone.utc)
    except Exception:
        last_dt = now

    elapsed_days = max((now - last_dt).total_seconds() / 86400.0, 0.001)
    stability = max(interval if interval is not None else 1, 1)

    # Retention formula: R = e^(-t / S)
    retention = math.exp(-elapsed_days / stability) * 100
    retention_pct = round(retention, 1)

    if retention_pct >= 80:
        status = "🟢 Strong Memory"
        color = "#00C853"
    elif retention_pct >= 50:
        status = "🟡 Medium Decay"
        color = "#FFD600"
    else:
        status = "🔴 High Decay (Review Recommended)"
        color = "#FF4B4B"

    return retention_pct, status, color

def update_card_review(card_id: str, quality: int, current_interval: int, current_ef: float, current_repetition: int):
    """Updates card memory parameters using SuperMemo-2 (SM-2)."""
    quality = max(0, min(5, quality))
    new_ef = current_ef + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
    new_ef = max(1.3, new_ef)

    if quality >= 3:
        if current_repetition == 0:
            new_interval = 1
        elif current_repetition == 1:
            new_interval = 6
        else:
            new_interval = round(current_interval * new_ef)
        new_repetition = current_repetition + 1
    else:
        new_repetition = 0
        new_interval = 1

    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    next_review = (datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=new_interval)).isoformat()

    supabase.table("flashcards").update({
        "interval": new_interval,
        "easiness_factor": new_ef,
        "repetition": new_repetition,
        "last_reviewed": now_iso,
        "next_review": next_review
    }).eq("id", card_id).execute()

# ==========================================
# 4. AUTHENTICATION & USER MANAGEMENT
# ==========================================

def login_user(username, password):
    res = supabase.table("users").select("*").eq("username", username).execute()
    if res.data:
        user = res.data[0]
        if bcrypt.checkpw(password.encode("utf-8"), user["password_hash"].encode("utf-8")):
            st.session_state.user = user
            st.rerun()
        else:
            st.error("Invalid password.")
    else:
        st.error("User not found.")

def register_user(username, password):
    res = supabase.table("users").select("*").eq("username", username).execute()
    if res.data:
        st.error("Username already taken.")
        return

    hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    new_user = supabase.table("users").insert({"username": username, "password_hash": hashed}).execute()
    if new_user.data:
        st.success("Account created successfully! Please log in.")

# Auth Screen
if not st.session_state.user:
    st.title("🧠 Multi-User AI Flashcard Hub")
    auth_tab1, auth_tab2 = st.tabs(["Login", "Register"])
    
    with auth_tab1:
        u = st.text_input("Username", key="login_u")
        p = st.text_input("Password", type="password", key="login_p")
        if st.button("Log In"):
            login_user(u, p)
            
    with auth_tab2:
        u_reg = st.text_input("Username", key="reg_u")
        p_reg = st.text_input("Password", type="password", key="reg_p")
        if st.button("Register Account"):
            register_user(u_reg, p_reg)
            
    st.stop()

# Logout in sidebar
st.sidebar.write(f"Logged in as: **{st.session_state.user['username']}**")
if st.sidebar.button("Log Out"):
    st.session_state.user = None
    st.rerun()

# ==========================================
# 5. MAIN APPLICATION TABS
# ==========================================

tab1, tab2, tab3 = st.tabs(["⚡ Generate Cards", "🎴 Smart Quiz", "📚 Dashboard & Retention"])

# ------------------------------------------
# TAB 1: GENERATE FLASHCARDS
# ------------------------------------------
with tab1:
    st.header("Generate AI Flashcards")
    subject = st.text_input("Subject / Topic", placeholder="e.g., Organic Chemistry, US History")
    input_text = st.text_area("Source Material or Notes", height=150)
    uploaded_file = st.file_uploader("Or Upload Document/Image", type=["txt", "pdf", "png", "jpg"])

    if st.button("Generate Cards"):
        content_payload = []
        if input_text:
            content_payload.append(input_text)
            
        if uploaded_file:
            bytes_data = uploaded_file.read()
            if uploaded_file.type == "text/plain":
                content_payload.append(bytes_data.decode("utf-8"))
            else:
                content_payload.append(
                    types.Part.from_bytes(data=bytes_data, mime_type=uploaded_file.type)
                )

        if not content_payload:
            st.warning("Please provide notes or upload a file.")
        else:
            with st.spinner("Analyzing content and building flashcards..."):
                schema = types.Schema(
                    type=types.Type.ARRAY,
                    items=types.Schema(
                        type=types.Type.OBJECT,
                        properties={
                            "question": types.Schema(type=types.Type.STRING),
                            "answer": types.Schema(type=types.Type.STRING)
                        },
                        required=["question", "answer"]
                    )
                )
                
                prompt = f"Create key study flashcards for subject '{subject}'. Extract key concepts into question and answer pairs."
                
                try:
                    raw_json = call_gemini_with_fallback(
                        prompt=[prompt] + content_payload,
                        response_schema=schema,
                        system_instruction="You are an expert tutor creating concise, accurate flashcards."
                    )
                    
                    cards = json.loads(raw_json)
                    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
                    
                    db_cards = []
                    for c in cards:
                        db_cards.append({
                            "user_id": st.session_state.user["id"],
                            "subject": subject or "General",
                            "question": c["question"],
                            "answer": c["answer"],
                            "interval": 1,
                            "easiness_factor": 2.5,
                            "repetition": 0,
                            "last_reviewed": now_iso,
                            "next_review": now_iso
                        })
                        
                    supabase.table("flashcards").insert(db_cards).execute()
                    st.success(f"Generated and saved {len(cards)} flashcards!")
                except Exception as e:
                    st.error(f"Failed to generate flashcards: {e}")

# ------------------------------------------
# TAB 2: SMART QUIZ (WITH RETENTION)
# ------------------------------------------
with tab2:
    st.header("Smart Quiz & Self-Evaluation")
    
    # Fetch cards for review
    res = supabase.table("flashcards").select("*").eq("user_id", st.session_state.user["id"]).execute()
    cards = res.data or []

    if not cards:
        st.info("No flashcards found. Create some in the 'Generate Cards' tab!")
    else:
        # Sort by lowest retention first (most urgently needing review)
        cards_with_retention = []
        for c in cards:
            ret, status, color = calculate_forgetting_curve(c.get("last_reviewed"), c.get("interval", 1))
            cards_with_retention.append((ret, c, status, color))
            
        cards_with_retention.sort(key=lambda x: x[0]) # Lowest retention first
        
        selected_card_tuple = cards_with_retention[0]
        retention, card, status, color = selected_card_tuple
        
        st.subheader(f"Subject: {card['subject']}")
        
        # Display Forgetting Curve Metric
        col_q, col_m = st.columns([3, 1])
        with col_m:
            st.metric("Estimated Memory Retention", f"{retention}%")
            st.markdown(f"<span style='color:{color}; font-weight:bold;'>{status}</span>", unsafe_allow_html=True)

        with col_q:
            st.markdown(f"### Q:")

        user_answer = st.text_area("Your Answer:", key=f"ans_{card['id']}")
        
        if st.button("Evaluate Answer"):
            if not user_answer:
                st.warning("Please enter an answer first.")
            else:
                with st.spinner("AI evaluating your response..."):
                    eval_schema = types.Schema(
                        type=types.Type.OBJECT,
                        properties={
                            "score": types.Schema(type=types.Type.INTEGER, description="Grade from 0 (completely wrong) to 5 (perfect execution)"),
                            "feedback": types.Schema(type=types.Type.STRING, description="Constructive feedback explaining the grade")
                        },
                        required=["score", "feedback"]
                    )
                    
                    eval_prompt = f"Correct Answer: {card['answer']}\nUser Answer: {user_answer}\nEvaluate accuracy and grade 0-5."
                    
                    eval_res = call_gemini_with_fallback(
                        prompt=eval_prompt,
                        response_schema=eval_schema,
                        system_instruction="You are an encouraging tutor grading student flashcard answers."
                    )
                    
                    eval_data = json.loads(eval_res)
                    score = eval_data["score"]
                    feedback = eval_data["feedback"]

                    st.markdown(f"**AI Grade:** {score}/5")
                    st.markdown(f"**Feedback:** {feedback}")
                    st.markdown(f"Actual Answer:")

                    # Update database with SM-2 algorithm
                    update_card_review(
                        card["id"],
                        score,
                        card.get("interval", 1),
                        card.get("easiness_factor", 2.5),
                        card.get("repetition", 0)
                    )
                    st.success("Card updated using Spaced Repetition + Forgetting Curve algorithm!")

# ------------------------------------------
# TAB 3: DASHBOARD & FORGETTING CURVE STATUS
# ------------------------------------------
with tab3:
    st.header("Deck Management & Retention Tracking")
    
    res = supabase.table("flashcards").select("*").eq("user_id", st.session_state.user["id"]).execute()
    cards = res.data or []
    
    if not cards:
        st.info("No saved flashcards.")
    else:
        table_data = []
        for c in cards:
            ret, status, _ = calculate_forgetting_curve(c.get("last_reviewed"), c.get("interval", 1))
            table_data.append({
                "ID": c["id"],
                "Subject": c["subject"],
                "Question": c["question"],
                "Retention %": f"{ret}%",
                "Status": status,
                "Interval (Days)": c.get("interval", 1),
                "Last Reviewed": c.get("last_reviewed", "Never")[:10]
            })

        df = pd.DataFrame(table_data)
        
        # Summary metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Cards", len(cards))
        avg_retention = round(sum([float(x["Retention %"].replace("%", "")) for x in table_data]) / len(cards), 1)
        m2.metric("Average Deck Retention", f"{avg_retention}%")
        critical_cards = len([x for x in table_data if "High Decay" in x["Status"]])
        m3.metric("Cards Needing Review", critical_cards)

        st.markdown("### Flashcard Inventory")
        st.dataframe(df.drop(columns=["ID"]), use_container_width=True)

        # Individual Card Deletion
        st.markdown("### Manage Cards")
        delete_id = st.selectbox("Select Card to Delete", options=[c["ID"] for c in table_data], format_func=lambda x: next(item["Question"] for item in table_data if item["ID"] == x))
        if st.button("Delete Selected Card"):
            supabase.table("flashcards").delete().eq("id", delete_id).execute()
            st.success("Card deleted successfully!")
            st.rerun()
