

def simple_chat_local(model, question):
    from langchain_ollama import OllamaLLM
    chat_model = OllamaLLM(model=model)
    return chat_model.invoke(question)