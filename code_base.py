import gradio as gr
import ollama
import warnings
warnings.filterwarnings("ignore")

MODEL = "qwen2.5-coder:3b"

simple_css = """
.gradio-container {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    min-height: 100vh;
    padding: 20px;
}
"""

system_prompt = (
    "You are a helpful assistant that provides concise, direct answers. "
    "Keep responses brief and to the point."
)

# I need to understand how this block works
def extract_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if block.get("type") == "text"
        )
    return ""


def chat_with_ollama(message, history):
    messages = [{"role": "system", "content": system_prompt}]

    for msg in history:
        text = extract_text(msg["content"])
        if text.strip():
            messages.append({
                "role": msg["role"],
                "content": text
            })

    messages.append({"role": "user", "content": message})

    stream = ollama.chat(
        model=MODEL,
        messages=messages,
        stream=True
    )

    partial = ""
    for chunk in stream:
        if "message" in chunk and "content" in chunk["message"]:
            partial += chunk["message"]["content"]
            yield partial



with gr.Blocks(
    theme=gr.themes.Soft(),
    css=simple_css
) as demo:

    gr.ChatInterface(
        chat_with_ollama,
        title=MODEL,
        description=f"Local AI assistant feat. {MODEL}",
        examples=[
            ["Hello!"],
            ["How LLMs work?"],
            ["Need help with Python code."]
        ],
        example_labels=[
            "👋 Greetings!",
            "🧠 Explain LLMs.",
            "🐍 Help me program."
        ],
        chatbot=gr.Chatbot(
            height=500,
            avatar_images=("llama logo.png", "qwen.png")
        )
    )

demo.launch()
