
import os
import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_astradb import AstraDBVectorStore
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import PydanticOutputParser
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.retrievers.contextual_compression import (
    ContextualCompressionRetriever,
)
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

# API keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ASTRA_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")

# Models
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=GOOGLE_API_KEY,
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0,
)

# Load PDFs
pdf_folder = "pdf" # create pdf folder  and add all file.pdf
documents = []

for file in os.listdir(pdf_folder):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(pdf_folder, file))
        docs = loader.load()

        for doc in docs:
            doc.metadata["source"] = file

        documents.extend(docs)

# Split documents
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

chunks = splitter.split_documents(documents)

# Store in AstraDB
vectorstore = AstraDBVectorStore(
    collection_name="multipdf_docs",
    embedding=embeddings,
    token=ASTRA_TOKEN,
    api_endpoint=ASTRA_ENDPOINT,
)

vectorstore.add_documents(chunks)

# MMR retriever
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 10},
)

# MultiQueryRetriever
multiquery_retriever = MultiQueryRetriever.from_llm(
    retriever=retriever,
    llm=llm,
)

# Contextual compression
compressor = LLMChainExtractor.from_llm(llm)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=multiquery_retriever,
)

# Pydantic output
class PDFAnswer(BaseModel):
    answer: str = Field(description="Answer based on the PDF context")
    not_found: bool = Field(
        description="True if the answer is not present in the context"
    )

    @field_validator("answer")
    @classmethod
    def clean_answer(cls, value):
        return value.strip()


parser = PydanticOutputParser(pydantic_object=PDFAnswer)

# Prompt
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        Answer the question using only the given PDF context.
        Use chat history only to understand the current question.
        Do not invent information.
        If the answer is unavailable, set not_found to true.
        Otherwise, set not_found to false.

        PDF Context:
        {context}

        {format_instructions}
        """,
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
]).partial(
    format_instructions=parser.get_format_instructions()
)

# Streamlit UI
st.title("Multi-PDF Analyzer")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    role = "user" if message.type == "human" else "assistant"
    with st.chat_message(role):
        st.write(message.content)

question = st.chat_input("Ask a question about your PDFs")

if question:
    # Retrieve documents
    retrieved_docs = compression_retriever.invoke(question)

    context = "\n\n".join(
        [
            f"Source: {doc.metadata.get('source', 'Unknown')}\n"
            f"Page: {doc.metadata.get('page', 0) + 1}\n"
            f"{doc.page_content}"
            for doc in retrieved_docs
        ]
    )

    # Generate answer
    messages = prompt.invoke({
        "context": context,
        "question": question,
        "chat_history": st.session_state.chat_history,
    })

    response = llm.invoke(messages)
    result = parser.invoke(response)

    # Display answer
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        st.write(result.answer)

        if result.not_found:
            st.write("Answer not found in the available PDF context.")

        with st.expander("Sources"):
            for doc in retrieved_docs:
                st.write(
                    f"PDF: {doc.metadata.get('source', 'Unknown')} | "
                    f"Page: {doc.metadata.get('page', 0) + 1}"
                )

    # Save chat history
    st.session_state.chat_history.extend([
        HumanMessage(content=question),
        AIMessage(content=result.answer),
    ])
