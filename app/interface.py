import gradio as gr
from chatbot.core import FAQRetriever

retreiever = FAQRetriever("data/faq.json")

# Main response function
def respond(message, chat_history):
    response = retreiever.get_answer(message)
    chat_history.append((message, response))
    return chat_history, ""


# Building UI using Blocks
with gr.Blocks() as demo:
    gr.Markdown("# 🤖 Chatbot Administratif Français")
    gr.Markdown("Posez des questions sur la CAF, CPAM, la préfecture, etc.")

    # chat display component
    chatbot = gr.Chatbot()

    # Input txt box for user messages
    msg = gr.Textbox(
        placeholder="Tapez votre question ici...",
        label="",
        container=True
    )

    examples = gr.Examples(
        examples=[
            "Comment faire une demande d'APL ?",
            "Quels documents pour renouveler un titre de séjour ?",
            "Comment obtenir une carte vitale ?"
        ],
        inputs=msg,
        fn=respond,
        outputs=[chatbot,msg],
        cache_examples=False,
        label="Examples rapides"
    )

    # Buttons: Submit and Clear history
    with gr.Row():
        submit_btn = gr.Button("Envoyer")
        clear_btn = gr.Button("Effacer")

    # Configuring interactions
    submit_btn.click(respond, [msg, chatbot], [chatbot, msg])
    msg.submit(respond, [msg, chatbot], [chatbot, msg])
    clear_btn.click(lambda: ([], ""), [], outputs=[chatbot,msg])
