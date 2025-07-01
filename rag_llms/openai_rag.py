import os

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


def load_env():
    import dotenv
    dotenv.load_dotenv()


def list_models():
    import openai
    openai.api_key = OPENAI_API_KEY
    models = openai.models.list()
    return [model.id for model in models.data]


def get_embedding_model(model_id):
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(model=model_id, api_key=OPENAI_API_KEY)


def get_chat_model(model_id):
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=model_id, api_key=OPENAI_API_KEY)


def embedding_distance(embedding_model, value1, value2):
    from langchain.evaluation import load_evaluator, EvaluatorType
    evaluator = load_evaluator(evaluator=EvaluatorType.EMBEDDING_DISTANCE, embeddings=embedding_model)
    return evaluator.evaluate_strings(prediction=value1, reference=value2)


def simple_chat(chat_model, question):
    return chat_model.invoke(input=question)


def extract(pdf_file):
    from langchain_community.document_loaders import PyPDFLoader
    loader = PyPDFLoader(file_path=pdf_file)
    return loader.load()


def chunk(documents):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000, chunk_overlap=200, length_function=len, separators=[' ', '\n', '\t'])
    return splitter.split_documents(documents=documents)


def delete_vectorstore(vectorstore_path):
    import shutil
    from pathlib import Path
    vectorstore = Path(vectorstore_path)
    if vectorstore.exists():
        shutil.rmtree(path=vectorstore)


def create_vectorstore(vectorstore_path, embedding_model, collection_name):
    from langchain_chroma import Chroma
    return Chroma(persist_directory=vectorstore_path, embedding_function=embedding_model, collection_name=collection_name)


def print_collections(vectorstore_path, embedding_model):
    from langchain_chroma import Chroma
    vectorstore = Chroma(persist_directory=vectorstore_path, embedding_function=embedding_model)
    client = vectorstore._client
    for collection in client.list_collections():
        name = collection.name
        collection = client.get_collection(name=name)
        print(f'Collection Name:{name} Entity Count:{collection.count()}')


def add_documents(vectorstore, documents):
    vectorstore.add_documents(documents=documents)


def get_documents(vectorstore, limit=1000, includes=None):
    if not includes:
        includes = ['metadatas', 'documents']
    return vectorstore._collection.get(limit=limit, include=includes)


def search_documents(vectorstore, question, count=5):
    return vectorstore.similarity_search(question, k=count)


def retriever_documents(vectorstore, question, count=5):
    retriever = vectorstore.as_retriever(search_type='similarity', search_kwargs={'k': count})
    return retriever.invoke(input=question)


def rag_chat(chat_model, vectorstore, prompt_template, question):
    from langchain_core.prompts import ChatPromptTemplate
    chat_prompt_template = ChatPromptTemplate.from_template(prompt_template)

    relevant_documents = retriever_documents(vectorstore, question)
    context_text = '\n\n---\n\n'.join(doc.page_content for doc in relevant_documents)

    prompt = chat_prompt_template.format(context=context_text, question=question)
    return chat_model.invoke(prompt)


def rag_chat_advanced(chat_model, vectorstore, prompt_template, question):
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough

    chat_prompt_template = ChatPromptTemplate.from_template(prompt_template)
    retriever = vectorstore.as_retriever(search_type='similarity')

    def context_text(relevant_documents):
        return '\n\n---\n\n'.join(doc.page_content for doc in relevant_documents)

    rag_chain = (
            {
                'context': retriever | context_text,
                'question': RunnablePassthrough()
            }
            | chat_prompt_template
            | chat_model
    )

    return rag_chain.invoke(input=question)


def rag_chat_app(pdf_file, vectorstore_path, question, advanced=False):
    documents = extract(pdf_file)
    chunks = chunk(documents)

    embedding_model = get_embedding_model('text-embedding-ada-002')
    vectorstore = create_vectorstore(vectorstore_path, embedding_model, f'rag_llms_{advanced}')
    add_documents(vectorstore, chunks)

    prompt_template = '''
            You are an assistant for question-answering tasks.
            Use the following pieces of retrieved context to answer the question. If you don't know the answer, say that you don't know. DON'T MAKE UP ANYTHING.

            {context}

            ---

            Answer the question based on the above context: {question}
            '''
    chat_model = get_chat_model('gpt-4o-mini')
    if advanced:
        return rag_chat_advanced(chat_model, vectorstore, prompt_template, question)
    return rag_chat(chat_model, vectorstore, prompt_template, question)


def direct_chat(chat_model, documents, prompt_template, question):
    from langchain_core.prompts import ChatPromptTemplate
    chat_prompt_template = ChatPromptTemplate.from_template(prompt_template)

    context_text = '\n\n---\n\n'.join(doc.page_content for doc in documents)
    prompt = chat_prompt_template.format(context=context_text, question=question)
    return chat_model.invoke(prompt)


def direct_chat_app(pdf_file, question):
    documents = extract(pdf_file)
    prompt_template = '''
            You are an assistant for question-answering tasks.
            Use the following pieces of retrieved context to answer the question. If you don't know the answer, say that you don't know. DON'T MAKE UP ANYTHING.

            {context}

            ---

            Answer the question based on the above context: {question}
            '''
    chat_model = get_chat_model('gpt-4o-mini')
    return direct_chat(chat_model, documents, prompt_template, question)


if __name__ == '__main__':
    load_env()
    print(f'Available Models: {list_models()}')

    _embedding_model = get_embedding_model('text-embedding-ada-002')
    print(f'Embedding Distance: {embedding_distance(_embedding_model, 'Amsterdam', 'coffee shop')}')

    _chat_model = get_chat_model('gpt-4o-mini')
    print(f'Chat Result: {simple_chat(_chat_model, 'Why did the chicken cross the road?')}')

    _pdf_path = '/Users/kuriankevin/Documents/GitHub/research-ai/rag_llms/data/Oppenheimer-2006-Applied_Cognitive_Psychology.pdf'
    _question = 'What is the title of the article?'

    _vectorstore_path = 'vectorstore'
    delete_vectorstore(_vectorstore_path)

    _rag_chat = rag_chat_app(_pdf_path, _vectorstore_path, _question)
    print(f'Rag Chat Response: {_rag_chat}')

    _rag_chat_chain = rag_chat_app(_pdf_path, _vectorstore_path, _question, advanced=True)
    print(f'Rag Chat Chain Response: {_rag_chat_chain}')

    print_collections(_vectorstore_path, _embedding_model)

    _direct_chat = direct_chat_app(_pdf_path, _question)
    print(f'Direct Chat Response: {_direct_chat}')
