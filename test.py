import streamlit as st
import folium
from langchain_core.callbacks import BaseCallbackHandler
from translate import Translator
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader, DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import Ollama
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.chains import RetrievalQA
from langchain.embeddings import OllamaEmbeddings
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

embeddings = OllamaEmbeddings()

if "page" not in st.session_state:
    st.session_state.page = "home"

st.sidebar.title("Chat Area")
chat_input = st.sidebar.text_input("Type a message here...", key="ollama_query")

if chat_input:
    vectordb = "chromadb"
    if chat_input:
        model = Ollama(
            model="tinyllama",
            callback_manager=CallbackManager([BaseCallbackHandler(), StreamingStdOutCallbackHandler()])
        )

        persist_directory = "chromadb"

        if not os.path.exists(persist_directory):
            with st.sidebar.spinner('🚀 Starting your bot.  This might take a while'):
                text_loader = DirectoryLoader("./data/", glob="./*.txt", loader_cls=TextLoader)

                text_documents = text_loader.load()

                splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=0)

                text_context = "\n\n".join(str(p.page_content) for p in text_documents)

                data = splitter.split_text(text_context)

                vectordb = Chroma.from_texts(data, embeddings, persist_directory=persist_directory)
                vectordb.persist()
        else:
            vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        query_chain = RetrievalQA.from_chain_type(llm="tinyllama", retriever=vectordb.as_retriever())
        prompt = """
                    As a Humanitarian Aid Worker, your primary goal is to assist refugees by addressing their questions
        promptly and clearly. Regardless of the language in which the queries are posed, strive to respond in 
        the same language for effective communication and understanding. Offer concise and professional 
        answers to ensure the refugees receive the support they need in a timely manner. Remember, your 
        responses should be both informative and empathetic to meet the refugees' diverse needs and situations
        Remember, your role is crucial in providing essential support and information to refugees in need. 
        Your professionalism and compassion can make a significant difference in their lives.
                    """
        response = query_chain({"query": prompt + chat_input})
        st.sidebar.write(f"Response: {response['result']}")


def main():
    if st.session_state.page == "home":
        home_page()
    elif st.session_state.page == "map":
        map_page()
    elif st.session_state.page == "translate":
        translate_page()


TRANSLATOR_LANGUAGES = ["en", "es", "fr", "de"]


def home_page():
    st.header("Welcome to the Refugee Assistance Hub")
    st.write("This platform provides resources and assistance tailored for refugees.")
    st.subheader("About Us")
    st.write("We are dedicated to supporting refugees worldwide by providing essential resources and information.")
    st.write("[Learn More](https://www.example.com/about)")
    st.subheader("Contact Us")
    st.write("For inquiries or support, please reach out to us at info@refugeehub.org")


def map_page():
    st.header("Refugee Camp Locations")
    location_data = {
        "Camp A": [40.7128, -74.0060],
        "Camp B": [34.0522, -118.2437]
    }
    m = folium.Map(location=[37.7749, -122.4194], zoom_start=5)
    for camp, coords in location_data.items():
        folium.Marker(coords, popup=f"{camp}<br>Capacity: 5000<br>Services: Food, Shelter").add_to(m)
    st.write("Click on markers for more details.")
    folium_static(m)


def folium_static(m):
    st.write("""
        <style>
.leaflet-container{
            height:600px;
            width:100%;
        }
        </style>
    """, unsafe_allow_html=True)
    st.write(m._repr_html_(), unsafe_allow_html=True)


def translate_page():
    st.header("Translation Service")
    translator = Translator(to_lang="fr")
    lang = st.selectbox("Select Language", TRANSLATOR_LANGUAGES, index=0)
    text = st.text_input("Enter text to translate:", "")
    if text:
        translator.to_lang = lang
        translated = translator.translate(text)
        st.write(f"Translated text: {translated}")


if __name__ == "__main__":
    page = st.sidebar.selectbox("Choose a page", ["Home", "Map", "Translate"])
    if page == "Home":
        home_page()
    elif page == "Map":
        map_page()
    elif page == "Translate":
        translate_page()
