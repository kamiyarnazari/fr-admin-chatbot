import gradio as gr
from chatbot.core import FAQRetriever

retreiever = FAQRetriever("data/faq.json")

def respond_to_user(message):
    return retreiever.get_answer(message)

# Gradio UI Setup
chatbot_UI = gr.Interface(
    fn=respond_to_user,
    inputs=gr.Textbox(label="Posez votre question ici"),
    outputs=gr.Textbox(label="Réponse du chatbot"),
    title="Chatbot Administratif Français",
    description="vous pouvez posez des questions sur la CAF, CPAM, préfecture, etc."
)

if __name__ == "__main__":
    chatbot_UI.launch()