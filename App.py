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
import resend

# ==========================================
# 1. INITIALIZATION & CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Multi-User AI Flashcard Hub",
    page_icon="🧠",
    layout="wide"
)

# Initialize Supabase
SUPABASE_URL = os.environ.get("SUPABASE_URL") or st.secrets.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY") or st.secrets.get("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("Missing Supabase configuration. Set `SUPABASE_URL` and `SUPABASE_KEY` environment variables or secrets.")
    st.stop()

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Initialize Email (Resend)
RESEND_API_KEY = os.environ.get("RESEND_API_KEY") or st.secrets.get("RESEND_API_KEY", None)
SENDER_EMAIL = os.environ.get("SENDER_EMAIL") or st.secrets.get("SENDER_EMAIL", "onboarding@resend.dev")

if RESEND_API_KEY:
    resend.api_key = RESEND_API_KEY

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
# 2. GEMINI & EMAIL HELPERS
# ==========================================

def get_gemini_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        st.error("Missing GEMINI_API_KEY environment variable or secret.")
        st.stop()
    return genai.Client(api_key=api_key)

def call_gemini_with_fallback(prompt, response_schema=None, system_instruction: str = None) -> str:
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
                # Silently log cooldown to session state without showing st.warning popups
                st.session_state.model_cooldowns[model] = now + datetime.timedelta(seconds=COOLDOWN_PERIOD_SECONDS)
                continue
            else:
                break

    st.error("All Gemini models are currently busy. Please wait a moment and try again.")
    return None

def send_study_reminder(user_email: str, username: str, due_count: int, critical_cards: list) -> bool:
    """Sends an email notification listing flashcards that are due for review."""
    if not RESEND_API_KEY:
        st.error("Resend API key is not configured.")
        return False

    card_items = "".join([f"<li><b>{c.get('subject', 'General')}</b>: {c.get('question', '')}</li>" for c in critical_cards[:5]])
    
    html_content = f"""
    <div style="font-family: Arial, sans-serif; padding: 20px; color: #333;">
        <h2>🧠 Flashcard Review Reminder</h2>
        <p>Hi <b>{username}</b>,</p>
        <p>You have <b>{due_count} flashcards</b> that are experiencing high memory decay and need review today to maintain retention!</p>
        
        <h3>Top Priority Cards:</h3>
        <ul>
            {card_items}
        </ul>
        
        <p>Open your Flashcard Hub to complete your review quiz!</p>
    </div>
    """

    try:
        params = {
            "from": SENDER_EMAIL,
            "to": [user_email],
            "subject": f"🧠 You have {due_count} flashcards due for review!",
            "html": html_content,
        }
        resend.Emails.send(params)
        return True
    except Exception as e:
        st.error(f"Failed to send email to {user_email}: {e}")
        return False

# ==========================================
# 3. FORGETTING CURVE & SM-2 LOGIC
# ==========================================

def calculate_forgetting_curve(last_reviewed: str, interval: int) -> tuple[float, str, str]:
    if not last_reviewed:
        return 0.0, "🔴 High Memory Decay (Needs Review)", "#FF4B4B"

    now = datetime.datetime.now(datetime.timezone.utc)
    
    try:
        clean_iso = str(last_reviewed).replace("Z", "+00:00")
        last_dt = datetime.datetime.fromisoformat(clean_iso)
        if last_dt.tzinfo is None:
            last_dt = last_dt.replace(tzinfo=datetime.timezone.utc)
        else:
            last_dt = last_dt.astimezone(datetime.timezone.utc)
    except Exception:
        last_dt = now

    elapsed_days = max((now - last_dt).total_seconds() / 86400.0, 0.001)
    stability = max(interval if interval is not None else 1, 1)

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
    quality = max(0, min(5, quality))
    
    ef = float(current_ef) if current_ef is not None else 2.5
    interval = int(current_interval) if current_interval is not None else 1
    repetition = int(current_repetition) if current_repetition is not None else 0

    new_ef = ef + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
    new_ef = max(1.3, new_ef)

    if quality >= 3:
        if repetition == 0:
            new_interval = 1
        elif repetition == 1:
            new_interval = 6
        else:
            new_interval = round(interval * new_ef)
        new_repetition = repetition + 1
    else:
        new_repetition = 0
        new_interval = 1

    now = datetime.datetime.now(datetime.timezone.utc)
    now_iso = now.isoformat()
    next_review_iso = (now + datetime.timedelta(days=int(new_interval))).isoformat()

    payload = {
        "interval": int(new_interval),
        "easiness_factor": float(round(new_ef, 2)),
        "repetition": int(new_repetition),
        "last_reviewed": now_iso,
        "next_review": next_review_iso
    }

    try:
        supabase.table("flashcards").update(payload).eq("id", card_id).execute()
    except Exception as e:
        st.error(f"Failed to update review status in database: {e}")

# ==========================================
# 4. AUTHENTICATION
# ==========================================

def login_user(username, password):
    if not username or not password:
        st.error("Please provide both username and password.")
        return

    try:
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
    except Exception as e:
        st.error("Database connection failed. Check your Supabase configuration.")

def register_user(username, password):
    res = supabase.table("users").select("*").eq("username", username).execute()
    if res.data:
        st.error("Username already taken.")
        return

    hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    new_user = supabase.table("users").insert({"username": username, "password_hash": hashed}).execute()
    if new_user.data:
        st.success("Account created successfully! Please log in.")

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

# Sidebar User Info & Logout
st.sidebar.write(f"Logged in as: **{st.session_state.user['username']}**")
if st.sidebar.button("Log Out"):
    st.session_state.user = None
    st.rerun()

# ==========================================
# 5. MAIN APPLICATION TABS
# ==========================================

tab1, tab2, tab3 = st.tabs(["⚡ Generate Cards", "🎴 Smart Quiz", "📚 Dashboard & Retention"])

# ------------------------------------------
# TAB 1: GENERATE & VIEW FLASHCARDS
# ------------------------------------------
with tab1:
    st.header("⚡ Generate & View Flashcards")
    
    subject = st.text_input("Subject / Topic", placeholder="e.g., Organic Chemistry, Cybersecurity")
    input_text = st.text_area("Source Material or Notes", height=150)
    uploaded_file = st.file_uploader("Or Upload Document/Image", type=["txt", "pdf", "png", "jpg"])

    if st.button("Generate Cards", type="primary"):
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
                    
                    if raw_json:
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
                        st.rerun()
                except Exception as e:
                    st.error(f"Failed to generate flashcards: {e}")

    st.divider()

    # Flashcard Inventory Table in Tab 1
    st.subheader("📋 Flashcard Inventory")
    res = supabase.table("flashcards").select("*").eq("user_id", st.session_state.user["id"]).execute()
    cards = res.data or []

    if not cards:
        st.info("No flashcards found. Use the generator above to create your first deck!")
    else:
        table_rows = []
        for c in cards:
            ret, status, _ = calculate_forgetting_curve(c.get("last_reviewed"), c.get("interval", 1))
            table_rows.append({
                "Subject": c.get("subject", "General"),
                "Question": c.get("question", ""),
                "Answer": c.get("answer", ""),
                "Retention": f"{ret}%",
                "Status": status
            })

        inventory_df = pd.DataFrame(table_rows)

        st.dataframe(
            inventory_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Subject": st.column_config.TextColumn("Subject", width="medium"),
                "Question": st.column_config.TextColumn("Question", width="large"),
                "Answer": st.column_config.TextColumn("Answer", width="large"),
                "Retention": st.column_config.TextColumn("Retention", width="small"),
                "Status": st.column_config.TextColumn("Status", width="medium"),
            }
        )

# ------------------------------------------
# TAB 2: SMART QUIZ
# ------------------------------------------
with tab2:
    st.header("🎴 Smart Quiz & Self-Evaluation")
    
    res = supabase.table("flashcards").select("*").eq("user_id", st.session_state.user["id"]).execute()
    cards = res.data or []

    if not cards:
        st.info("No flashcards found. Create some in the 'Generate Cards' tab!")
    else:
        cards_with_retention = []
        for c in cards:
            ret, status, color = calculate_forgetting_curve(c.get("last_reviewed"), c.get("interval", 1))
            cards_with_retention.append((ret, c, status, color))
            
        cards_with_retention.sort(key=lambda x: x[0])  # Lowest retention first

        if "card_index" not in st.session_state or st.session_state.card_index >= len(cards_with_retention):
            st.session_state.card_index = 0

        retention, card, status, color = cards_with_retention[st.session_state.card_index]

        col_header, col_nav = st.columns([3, 1])
        with col_header:
            st.caption(f"Currently Reviewing: Card {st.session_state.card_index + 1} of {len(cards_with_retention)}")
            st.subheader(f"Subject: {card['subject']}")
        with col_nav:
            if st.button("⏭️ Skip to Next Card"):
                st.session_state.card_index = (st.session_state.card_index + 1) % len(cards_with_retention)
                st.rerun()

        st.divider()

        col_q, col_m = st.columns([3, 1])
        with col_q:
            st.markdown(f"### Q: {card['question']}")
        with col_m:
            st.metric("Memory Retention", f"{retention}%")
            st.markdown(f"<span style='color:{color}; font-weight:bold;'>{status}</span>", unsafe_allow_html=True)

        st.divider()

        user_answer = st.text_area("Your Answer:", key=f"ans_{card['id']}", height=120)

        col_btn1, col_btn2 = st.columns([1, 1])
        with col_btn1:
            eval_clicked = st.button("🤖 Evaluate Answer with AI", type="primary", use_container_width=True)
        with col_btn2:
            show_answer = st.button("👁️ Reveal Correct Answer", use_container_width=True)

        if show_answer:
            st.info(f"Correct Answer:")

       # AI Evaluation Flow
        if eval_clicked:
            if not user_answer.strip():
                st.warning("Please type an answer before requesting AI feedback.")
            else:
                # Retrieve answer safely
                correct_answer = card.get("answer") or card.get("Answer") or "No answer specified for this card."

                with st.spinner("AI evaluating your response..."):
                    eval_schema = types.Schema(
                        type=types.Type.OBJECT,
                        properties={
                            "score": types.Schema(type=types.Type.INTEGER, description="Grade from 0 to 5"),
                            "feedback": types.Schema(type=types.Type.STRING, description="Feedback explaining the score")
                        },
                        required=["score", "feedback"]
                    )
                    
                    eval_prompt = f"Correct Answer: {correct_answer}\nUser Answer: {user_answer}\nEvaluate accuracy and grade 0-5."
                    
                    eval_res = call_gemini_with_fallback(
                        prompt=eval_prompt,
                        response_schema=eval_schema,
                        system_instruction="You are an encouraging tutor grading student flashcard answers."
                    )

                    if eval_res:
                        eval_data = json.loads(eval_res)
                        score = eval_data.get("score", 0)
                        feedback = eval_data.get("feedback", "")

                        st.markdown("### AI Evaluation Result")
                        if score >= 4:
                            st.success(f"**Grade: {score}/5** — Excellent job!")
                        elif score >= 2:
                            st.warning(f"**Grade: {score}/5** — Getting there!")
                        else:
                            st.error(f"**Grade: {score}/5** — Needs review.")

                        st.write(f"**Feedback:** {feedback}")
                        st.info(f"**Expected Answer:** {correct_answer}")

                        update_card_review(
                            card["id"],
                            score,
                            card.get("interval", 1),
                            card.get("easiness_factor", 2.5),
                            card.get("repetition", 0)
                        )
                        st.success("✅ Card memory schedule updated!")

                        if st.button("Continue to Next Card ➔"):
                            st.session_state.card_index = (st.session_state.card_index + 1) % len(cards_with_retention)
                            st.rerun()

# ------------------------------------------
# TAB 3: DASHBOARD & RETENTION
# ------------------------------------------
with tab3:
    st.header("📚 Dashboard & Retention Tracking")
    
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
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Cards", len(cards))
        avg_retention = round(sum([float(x["Retention %"].replace("%", "")) for x in table_data]) / len(cards), 1)
        m2.metric("Average Deck Retention", f"{avg_retention}%")
        critical_cards = len([x for x in table_data if "High Decay" in x["Status"]])
        m3.metric("Cards Needing Review", critical_cards)

        st.markdown("### Flashcard Inventory")
        st.dataframe(df.drop(columns=["ID"]), use_container_width=True)

        st.divider()

        # Email Notifications Section
        st.subheader("📧 Email Notifications")
        user_email = st.text_input("Notification Email Address", value=st.session_state.user.get("email", ""))

        if st.button("Save Email & Send Review Summary"):
            if not user_email:
                st.warning("Please enter a valid email address.")
            else:
                supabase.table("users").update({"email": user_email}).eq("id", st.session_state.user["id"]).execute()
                st.session_state.user["email"] = user_email
                
                due_cards = [c for c in cards if calculate_forgetting_curve(c.get("last_reviewed"), c.get("interval", 1))[0] < 50]
                
                if due_cards:
                    success = send_study_reminder(user_email, st.session_state.user["username"], len(due_cards), due_cards)
                    if success:
                        st.success(f"Review digest sent to {user_email}!")
                else:
                    st.info("Your retention is strong across all cards! No digest needed right now.")

        st.divider()

        # Manage Cards Section
        st.markdown("### Manage Cards")
        delete_id = st.selectbox(
            "Select Card to Delete",
            options=[c["ID"] for c in table_data],
            format_func=lambda x: next(item["Question"] for item in table_data if item["ID"] == x)
        )
        if st.button("Delete Selected Card"):
            supabase.table("flashcards").delete().eq("id", delete_id).execute()
            st.success("Card deleted successfully!")
            st.rerun()
