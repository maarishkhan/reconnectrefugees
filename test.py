import streamlit as st
import folium
from streamlit_folium import st_folium
from translate import Translator
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.embeddings import OllamaEmbeddings
from dotenv import load_dotenv
import os
import subprocess

load_dotenv()

embeddings = OllamaEmbeddings()

if "page" not in st.session_state:
    st.session_state.page = "home"

st.sidebar.title("Chat Area")
chat_input = st.sidebar.text_input("Type a message here...", key="ollama_query")


def check_model_availability(Mname):
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        if result.returncode == 0:
            models_list = result.stdout
            return Mname in models_list
        else:
            st.sidebar.error("Error checking available models.")
            return False
    except Exception as e:
        st.sidebar.error(f"An error occurred while checking model availability: {e}")
        return False


prompt = """
        As a Humanitarian Aid Worker, your primary goal is to assist refugees by addressing their questions
        promptly and clearly. Regardless of the language in which the queries are posed, strive to respond in 
        the same language for effective communication and understanding. Offer concise and professional 
        answers to ensure the refugees receive the support they need in a timely manner. Remember, your 
        responses should be both informative and empathetic to meet the refugees' diverse needs and situations.
        Remember, your role is crucial in providing essential support and information to refugees in need. 
        Your professionalism and compassion can make a significant difference in their lives.
        """
model_name = "tinyllama:latest"
model = Ollama(model=model_name)
persist_directory = "chromadb"
data_directory = "./data/"
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
retriever = vectordb.as_retriever()
query_chain = RetrievalQA.from_chain_type(llm=model, retriever=retriever, chain_type="stuff")

try:
    response = query_chain({"query": prompt})
    st.sidebar.write(f"Response: {response['result']}")
except ValueError as e:
    st.sidebar.error(f"Error in processing the query: {e}")

if chat_input:
    model_name = "tinyllama:latest"
    if not check_model_availability(model_name):
        st.sidebar.error(f"Model '{model_name}' not found. Please ensure it is available.")
    else:
        model = Ollama(model=model_name)

        persist_directory = "chromadb"
        data_directory = "./data/"
        if not os.path.exists(persist_directory):
            if os.path.exists(data_directory):
                with st.spinner('🚀 Starting your bot. This might take a while'):
                    text_loader = DirectoryLoader(data_directory, glob="./*.txt", loader_cls=TextLoader)
                    text_documents = text_loader.load()
                    splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=5)
                    text_context = "\n\n".join(str(p.page_content) for p in text_documents)
                    data = splitter.split_text(text_context)
                    vectordb = Chroma.from_texts(data, embeddings, persist_directory=persist_directory)
                    vectordb.persist()
            else:
                st.sidebar.error(
                    f"Data directory '{data_directory}' not found. Please ensure it exists and contains the necessary "
                    f"text files.")
        else:
            vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

        retriever = vectordb.as_retriever()
        query_chain = RetrievalQA.from_chain_type(llm=model, retriever=retriever, chain_type="stuff")

        try:
            response = query_chain({"query": chat_input})
            st.sidebar.write(f"Response: {response['result']}")
        except ValueError as e:
            st.sidebar.error(f"Error in processing the query: {e}")
            st.write(text_documents)


def main():
    if st.session_state.page == "home":
        home_page()
    elif st.session_state.page == "map":
        map_page()
    elif st.session_state.page == "translate":
        translate_page()


TRANSLATOR_LANGUAGES = ["en", "es", "fr", "de", "ar", "it", "ru", "ja", "zh"]


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
    st_data = st_folium(m, width=725)


def translate_page():
    st.header("Translation Service")
    lang = st.selectbox("Select Language", TRANSLATOR_LANGUAGES, index=0)
    text = st.text_input("Enter text to translate:", "")
    if text:
        translator = Translator(to_lang=lang)
        final = translator.translate(text)
        st.write(f"Translated text: {final}")


if __name__ == "__main__":
    page = st.sidebar.selectbox("Choose a page", ["Home", "Map", "Translate"])
    if page == "Home":
        home_page()
    elif page == "Map":
        map_page()
    elif page == "Translate":
        translate_page()
