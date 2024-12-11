import os
import re
import openai
import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
import constants

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = constants.OPEN_API_KEY

# File for storing the data
DATA_FILE = "Data.txt"
PERSIST = False

def update_data_file(old_text, new_text):
    """Replaces old_text with new_text in the data file and rebuilds the vectorstore."""
    try:
        with open(DATA_FILE, "r") as f:
            file_content = f.read()
        updated_content = file_content.replace(old_text, new_text)
        with open(DATA_FILE, "w") as f:
            f.write(updated_content)
    except FileNotFoundError:
        st.error("Error: Data.txt file not found. Please ensure the file exists.")
        return None
    except Exception as e:
        st.error(f"Error updating file: {e}")
        return None

    # Rebuild the vectorstore
    try:
        loader = TextLoader(DATA_FILE)
        index_kwargs = {"persist_directory": "persist"} if PERSIST else {}
        embeddings = OpenAIEmbeddings()  # Ensure embedding consistency
        vectorstore = Chroma.from_documents(loader.load(), embeddings, **index_kwargs)
        return VectorStoreIndexWrapper(vectorstore=vectorstore)
    except Exception as e:
        st.error(f"Error rebuilding the index: {e}")
        return None

def create_chain(index):
    """Creates a conversational chain using the index."""
    retriever = index.vectorstore.as_retriever(search_kwargs={"k": 1})
    return ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model="gpt-3.5-turbo"),
        retriever=retriever,
        return_source_documents=False
    )

# Load or create the index
if PERSIST and os.path.exists("persist"):
    st.write("Reusing existing index...\n")
    vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
    index = VectorStoreIndexWrapper(vectorstore=vectorstore)
else:
    loader = TextLoader(DATA_FILE)
    index_kwargs = {"persist_directory": "persist"} if PERSIST else {}
    embeddings = OpenAIEmbeddings()  # Initialize OpenAIEmbeddings
    documents = loader.load()
    vectorstore = Chroma.from_documents(documents, embeddings, **index_kwargs)
    index = VectorStoreIndexWrapper(vectorstore=vectorstore)

# Initialize conversational chain
chain = create_chain(index)

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# UI for the chatbot
st.title("Romario's Agent")

# Input for the user query
query = st.text_input("Enter your prompt or question:", key="query")

if st.button("Send"):
    if query:
        try:
            # Attempt to get a response from the retrieval chain
            result = chain({"question": query, "chat_history": st.session_state["chat_history"]})
            bot_response = result.get("answer", "").strip()
            
            # If no relevant information found in the index, fallback to LLM
            if not bot_response or "I don't know" in bot_response:
                llm = ChatOpenAI(model="gpt-3.5-turbo")  # Use OpenAI for general queries
                bot_response = llm.predict(query)
            
            # Update chat history
            st.session_state["chat_history"].append((query, bot_response))
            st.write("**You:**", query)
            st.write("**Bot:**", bot_response)
        except KeyError:
            st.error("Unexpected response format. Please check chain configuration.")
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Please enter a prompt or question.")

# Display chat history
st.subheader("Chat History")
for idx, (user_query, bot_response) in enumerate(st.session_state["chat_history"]):
    st.write(f"**Prompt {idx + 1}:** {user_query}")
    st.write(f"**Answer {idx + 1}:** {bot_response}")
    st.write("---")
