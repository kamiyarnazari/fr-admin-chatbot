from chatbot.core import FAQRetriever

retriever = FAQRetriever("data/faq.json")

while True:
    query = input("vous: ")
    if query.lower() in ["exit", "quit"]:
        break
    response = retriever.get_answer(query)
    print("Bot:", response)