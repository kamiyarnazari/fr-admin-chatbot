import gradio as gr
from chatbot.retriever import FAQWithLLM

"""
UI module for French administration chatbot. Uses Gradio for user interaction.
"""


retriever = FAQWithLLM("data/faq.json")

# Main response function
def respond(message, chat_history):
    print(f"📨 Question reçue: {message}") 
    if chat_history is None:
        chat_history = []

    answer = retriever.get_answer(message, chat_history)

    chat_history.append({"role": "user",      "content": message})
    chat_history.append({"role": "assistant", "content": answer})

    return chat_history, ""


# Building UI using Blocks
with gr.Blocks() as demo:
    gr.Markdown("# 🤖 Chatbot Administratif Français")
    gr.Markdown("Posez des questions sur la CAF, CPAM, la préfecture, etc.")
    # chat display component
    chatbot = gr.Chatbot(type="messages")

    # Input txt box for user messages
    msg = gr.Textbox(
        placeholder="Tapez votre question ici...",
        label="",
        container=True
    )

    gr.Markdown("📌 *Projet open-source développé par [Kamyar](https://github.com/kamiyarnazari)*", elem_id="footer")


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

