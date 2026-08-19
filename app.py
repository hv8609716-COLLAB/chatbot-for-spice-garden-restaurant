import os
import gradio as gr
from huggingface_hub import InferenceClient

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

DEFAULT_MODEL = "Qwen 2.5 7B (Fast)"
SHOW_MODEL_SWITCH = False

PRIMARY_COLOR = "orange"

# ============================================================
# ⚙️ CORE ENGINE — no need to touch below this line
# ============================================================

HF_TOKEN = os.environ.get("HF_TOKEN")

MODELS = {
    "Qwen 2.5 7B (Fast)": "Qwen/Qwen2.5-7B-Instruct",
    "Llama 3.1 8B (Smart)": "meta-llama/Llama-3.1-8B-Instruct",
}


def chat(message, history, model_choice):
    model_name = model_choice if SHOW_MODEL_SWITCH else DEFAULT_MODEL
    client = InferenceClient(model=MODELS[model_name], token=HF_TOKEN)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for item in history:
        if isinstance(item, dict):
            messages.append({"role": item["role"], "content": item["content"]})
        else:
            user_msg, bot_msg = item
            messages.append({"role": "user", "content": user_msg})
            if bot_msg:
                messages.append({"role": "assistant", "content": bot_msg})

    messages.append({"role": "user", "content": message})

    response = ""
    try:
        stream = client.chat_completion(
            messages=messages,
            max_tokens=512,
            temperature=0.4,
            top_p=0.9,
            stream=True,
        )
        for chunk in stream:
            if chunk.choices and len(chunk.choices) > 0:
                token = chunk.choices[0].delta.content
                if token:
                    response += token
                    yield response

        if not response:
            yield "⚠️ The assistant didn't return a response. Please try again."

    except Exception as e:
        yield f"⚠️ Something went wrong. Please try again in a moment."


custom_css = """
#component-0 { max-width: 900px; margin: auto; }
.gradio-container { background: linear-gradient(135deg, #1e1b2e 0%, #2d2640 100%); }
footer { visibility: hidden; }
"""

with gr.Blocks(title=BOT_NAME) as demo:

    gr.Markdown(
        f"""
        <div style="text-align:center;">
        <h1>🍽️ {BOT_NAME}</h1>
        <p style="color:gray;">{BOT_TAGLINE}</p>
        </div>
        """
    )

    model_choice = gr.State(DEFAULT_MODEL)

    gr.ChatInterface(
        fn=chat,
        additional_inputs=[model_choice],
        chatbot=gr.Chatbot(
            height=520,
            label="Conversation",
            avatar_images=(None, "https://em-content.zobj.net/source/microsoft-teams/363/robot_1f916.png"),
        ),
        title=None,
    )

    gr.Markdown(
        f"""
        <p style="text-align:center; color:gray; font-size:12px; margin-top:20px;">
        Demo built by Harsh · Custom AI Chatbots for Businesses 🚀
        </p>
        """
    )

demo.launch(theme=gr.themes.Soft(primary_hue=PRIMARY_COLOR), css=custom_css)  
