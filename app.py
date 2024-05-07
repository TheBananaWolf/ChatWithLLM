import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.chains.conversation.base import ConversationChain
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import Ollama
from htmlTemplates import css, bot_template, user_template
from langchain import PromptTemplate


def getPdfText(pdfDocs):
    text = ""
    for pdf in pdfDocs:
        pdfReader = PdfReader(pdf)
        for page in pdfReader.pages:
            text += page.extract_text()
    return text


def getTextChunk(pdf):
    textSpliter = CharacterTextSplitter("\n", chunk_size=1000, chunk_overlap=300, length_function=len)
    chunk = textSpliter.split_text(pdf)
    return chunk


def getVectorStore(chunk):
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorStore = FAISS.from_texts(chunk, embeddings)
    return vectorStore


def getConversationChain(vectorStore):
    # repo_id = "deepset/roberta-base-squad2"

    # llm = HuggingFaceEndpoint(
    #     repo_id=repo_id, max_length=512, temperature=0.5
    # )
    llm = Ollama(model="llama3")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversationChain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorStore.as_retriever(),
        memory=memory
    )
    return conversationChain


def getConversationChainForChat():
    # repo_id = "deepset/roberta-base-squad2"

    # llm = HuggingFaceEndpoint(
    #     repo_id=repo_id, max_length=512, temperature=0.5
    # )
    llm = Ollama(model="llama3")
    conversationChain = ConversationChain(
        llm=llm,
        memory=ConversationBufferMemory(memory_key="history", return_messages=True)
    )
    return conversationChain


def handleQuestionUsingSelf(question):
    response = st.session_state.self(question)



    st.session_state.chat_history = response['history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.chat_message("user").write(message.content)
        else:
            st.chat_message("assistant").write(message.content)


def handleQuestion(question):
    response = st.session_state.conversation(question)
    prompt_template = """
        You are an advisor. Based on the question: '{question}', please optimize this response: '{response}'. 
        If the response is 'I don't know' or something similar, use your knowledge to answer the question.
    """
    prompt = PromptTemplate.from_template(template=prompt_template)
    prompt_formatted_str: str = prompt.format(
       question=question, response=response)

    llm = Ollama(model="llama3")
    prediction = llm.invoke(prompt_formatted_str)

    st.session_state.chat_history.append({"type": "Human", "content": question})
    st.session_state.chat_history.append({"type": "AI", "content": prediction})

    for message in st.session_state.chat_history:
        if message['type'] == "Human":
            st.chat_message("Human").write(message['content'])
        else:
            st.chat_message("Assistant").write(message['content'])

def main():
    load_dotenv()

    st.set_page_config(page_title="Your GPTs", page_icon=":books")
    st.write(css, unsafe_allow_html=True)

    st.header("Your Free Local GPTs")
    question = st.chat_input("Talk with me anything you want to know")

    # if "mode" not in st.session_state:
    #     # false using self knowledge
    #     st.session_state.mode = False
    #
    #
    # if str(question).lower() == "pdf":
    #     st.session_state.mode = True
    #     st.info("switch mode, please upload files.")
    # if str(question).lower() == "not pdf":
    #     st.session_state.mode = False
    #     st.info("switch mode")

    # if question != None and question != "" and st.session_state.mode == False:
    #     if "self" not in st.session_state:
    #         st.session_state.self = getConversationChainForChat()
    #     handleQuestionUsingSelf(question)
    #
    # if question != None and question != "" and st.session_state.mode == True:
    #     if "conversation" not in st.session_state:
    #         st.session_state.conversation = None
    #     elif "vectorStore" not in st.session_state:
    #         st.info("Please upload files.")
    #     else:
    if question != None and question != "":
        if "conversation" not in st.session_state:
            st.session_state.conversation = None
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        handleQuestion(question)

    with st.sidebar:
        st.subheader("Your Files")
        pdfDocs = st.file_uploader("Upload your files", accept_multiple_files=True)
        if st.button("Upload"):
            with st.spinner("Process"):
                # st.session_state.mode == True

                rawText = getPdfText(pdfDocs)

                textChunk = getTextChunk(rawText)

                st.session_state.vectorStore = getVectorStore(textChunk)

                st.session_state.conversation = getConversationChain(st.session_state.vectorStore)


if __name__ == '__main__':
    main()
