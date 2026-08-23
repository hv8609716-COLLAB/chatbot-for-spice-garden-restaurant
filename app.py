import os
import streamlit as st
from groq import Groq

# ============================================================
# 🔧 CUSTOMIZE THIS SECTION FOR EACH CLIENT — nothing else needs to change
# ============================================================

BOT_NAME = "Spice Garden Assistant"
BOT_TAGLINE = "Your friendly restaurant helper · Ask about menu, timings & bookings"

SYSTEM_PROMPT = """You are the official AI assistant for "Spice Garden", a family-friendly Indian restaurant.

Restaurant details:
- Location: MG Road, Delhi
- Timings: 11 AM to 11 PM, all days
- Cuisine: North Indian, Chinese, and Continental
- Popular dishes: Butter Chicken, Paneer Tikka, Veg Biryani, Chocolate Lava Cake
- Home delivery available via Zomato and Swiggy
- Table booking: customers can call +91-XXXXXXXXXX or book via this chat
- Average cost for two: ₹800

Your job:
- Answer customer questions about the menu, timings, pricing, and booking
- Be warm, polite, and helpful — like a friendly restaurant host
- If asked something you don't know, politely suggest calling the restaurant directly
- Always respond in the same language the user writes in (Hindi, English, or Hinglish)
- Keep responses concise and natural, avoid sounding robotic
"""

DEFAULT_MODEL = "openai/gpt-oss-120b"
SHOW_MODEL_SWITCH = False

PRIMARY_COLOR = "#FF7A18"   # orange, used for accents in the CSS below

# ============================================================
# ⚙️ CORE ENGINE — no need to touch below this line
# ============================================================

GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY"))

# 👉 Agar model switch dikhana hai to yaha aur model names add kar sakta hai
MODELS = {
    "your-model-name-here": "your-model-name-here",
}

st.set_page_config(page_title=BOT_NAME, page_icon="🍽️", layout="centered")

# ---------- Light theming via CSS ----------
st.markdown(
    f"""
    <style>
    .main {{
        background: linear-gradient(135deg, #1e1b2e 0%, #2d2640 100%);
    }}
    #MainMenu, footer {{visibility: hidden;}}
    .bot-title {{
        text-align: center;
    }}
    .bot-title h1 {{
        margin-bottom: 0;
    }}
    .bot-title p {{
        color: gray;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_client():
    return Groq(api_key=GROQ_API_KEY)


# ---------- Session state ----------
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": ..., "content": ...}
if "model_choice" not in st.session_state:
    st.session_state.model_choice = DEFAULT_MODEL


def stream_chat(message, history, model_choice):
    model_name = model_choice if SHOW_MODEL_SWITCH else DEFAULT_MODEL
    client = get_client()

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for item in history:
        messages.append({"role": item["role"], "content": item["content"]})
    messages.append({"role": "user", "content": message})

    try:
        stream = client.chat.completions.create(
            model=MODELS.get(model_name, model_name),
            messages=messages,
            max_tokens=512,
            temperature=0.4,
            top_p=0.9,
            stream=True,
        )
        response = ""
        for chunk in stream:
            if chunk.choices and len(chunk.choices) > 0:
                token = chunk.choices[0].delta.content
                if token:
                    response += token
                    yield response

        if not response:
            yield "⚠️ The assistant didn't return a response. Please try again."

    except Exception:
        yield "⚠️ Something went wrong. Please try again in a moment."


# ---------- UI ----------
st.markdown(
    f"""
    <div class="bot-title">
    <h1>🍽️ {BOT_NAME}</h1>
    <p>{BOT_TAGLINE}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if SHOW_MODEL_SWITCH:
    st.session_state.model_choice = st.selectbox(
        "Model", list(MODELS.keys()), index=list(MODELS.keys()).index(st.session_state.model_choice)
    )

# Display existing chat history
for msg in st.session_state.messages:
    avatar = "🍽️" if msg["role"] == "assistant" else None
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# Chat input
user_message = st.chat_input("Ask about the menu, timings, or booking...")

if user_message:
    st.session_state.messages.append({"role": "user", "content": user_message})
    with st.chat_message("user"):
        st.markdown(user_message)

    with st.chat_message("assistant", avatar="🍽️"):
        placeholder = st.empty()
        full_response = ""
        for partial in stream_chat(user_message, st.session_state.messages[:-1], st.session_state.model_choice):
            full_response = partial
            placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})

st.markdown(
    """
    <p style="text-align:center; color:gray; font-size:12px; margin-top:20px;">
    Demo built by Harsh · Custom AI Chatbots for Businesses 🚀
    </p>
    """,
    unsafe_allow_html=True,
)
