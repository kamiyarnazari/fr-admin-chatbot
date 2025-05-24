import gradio as gr
from chatbot.core import FAQRetriever

retreiever = FAQRetriever("data/faq.json")

def respond_to_user(message):
    return retreiever.get_answer(message)

# Gradio UI Setup
chatbot_UI = gr.Interface(
    fn=respond_to_user,
    inputs=gr.Textbox(lines=2, label="Posez votre question ici"),
    outputs=gr.Markdown(label="Réponses proposées"),
    title="Chatbot Administratif Français",
    description="Voici les réponses les plus pertinentes basées sur votre question."
)

if __name__ == "__main__":
    chatbot_UI.launch()