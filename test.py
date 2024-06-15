import os
import subprocess
import streamlit as st
import folium
from streamlit_folium import st_folium
from translate import Translator
from langchain.vectorstores import Chroma
from langchain.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.embeddings import OllamaEmbeddings
from dotenv import load_dotenv
from streamlit_option_menu import option_menu

load_dotenv()
embeddings = OllamaEmbeddings()

if "history" not in st.session_state:
    st.session_state.history = []

if "language" not in st.session_state:
    st.session_state.language = "en"

st.set_page_config(page_title="Refugee Assistance Hub", page_icon="🌍")

MODEL_NAME = "tinyllama:latest"
PERSIST_DIRECTORY = "chromadb"
DATA_DIRECTORY = "./data/"


def check_model_availability(Mname):
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        return result.returncode == 0 and Mname in result.stdout
    except Exception as e:
        st.error(f"Error checking model availability: {e}")
        return False


def initialize_model_and_db():
    if not check_model_availability(MODEL_NAME):
        st.error(f"Model '{MODEL_NAME}' not found.")
        return None, None
    llm = Ollama(model=MODEL_NAME)
    if not os.path.exists(PERSIST_DIRECTORY):
        if os.path.exists(DATA_DIRECTORY):
            with st.spinner('🚀 Starting your bot. This might take a while'):
                pdf_loader = DirectoryLoader(DATA_DIRECTORY, glob="*.pdf", loader_cls=PyPDFLoader)
                text_loader = DirectoryLoader(DATA_DIRECTORY, glob="*.txt", loader_cls=TextLoader)
                pdf_documents = pdf_loader.load()
                text_documents = text_loader.load()
                splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=0)
                pdf_context = "\n\n".join(str(p.page_content) for p in pdf_documents)
                text_context = "\n\n".join(str(p.page_content) for p in text_documents)
                pdfs = splitter.split_text(pdf_context)
                texts = splitter.split_text(text_context)
                data = pdfs + texts
                db = Chroma.from_texts(data, embeddings, persist_directory=PERSIST_DIRECTORY)
                db.persist()
        else:
            st.error(f"Data directory '{DATA_DIRECTORY}' not found.")
            return None, None
    else:
        db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
    return llm, db


def get_initial_response():
    prompt = """
    As a Humanitarian Aid Worker, your primary goal is to assist refugees by addressing their questions
    promptly and clearly. Regardless of the language in which the queries are posed, strive to respond in 
    the same language for effective communication and understanding. Offer concise and professional 
    answers to ensure the refugees receive the support they need in a timely manner. Remember, your 
    responses should be both informative and empathetic to meet the refugees' diverse needs and situations.
    Remember, your role is crucial in providing essential support and information to refugees in need. 
    Your professionalism and compassion can make a significant difference in their lives.
    """
    llm, db = initialize_model_and_db()
    if llm and db:
        retriever = db.as_retriever()
        chat_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")
        try:
            result = chat_chain({"query": prompt})
            st.write(f"Response: {result['result']}")
        except ValueError as err:
            st.error(f"Error processing the query: {err}")


get_initial_response()


def translate_text(text, to_lang):
    if to_lang == "en":
        return text
    translator = Translator(to_lang=to_lang)
    try:
        return translator.translate(text)
    except Exception as e:
        st.error(f"Translation error: {e}")
        return text


def home_page():
    st.header(translate_text("Welcome to the Refugee Assistance Hub", st.session_state.language))
    st.write(translate_text("This platform provides resources and assistance tailored for refugees.",
                            st.session_state.language))
    st.subheader(translate_text("About Us", st.session_state.language))
    st.write(translate_text(
        "We are dedicated to supporting refugees worldwide by providing essential resources and information.",
        st.session_state.language))
    st.write("[Learn More](https://www.example.com/about)")
    st.subheader(translate_text("Contact Us", st.session_state.language))
    st.write(translate_text("For inquiries or support, please reach out to us at info@refugeehub.org",
                            st.session_state.language))


def map_page():
    st.header(translate_text("Refugee Camp Locations", st.session_state.language))
    location_data = {
        "Camp A": [40.7128, -74.0060],
        "Camp B": [34.0522, -118.2437],
        "Camp C": [37.7749, -122.4194],
        "Camp D": [48.8566, 2.3522],
        "Camp E": [35.6895, 139.6917]
    }
    m = folium.Map(location=[0, 0], zoom_start=2)
    for camp, coords in location_data.items():
        folium.Marker(
            coords,
            popup=f"{camp}<br>{translate_text('Capacity', st.session_state.language)}: 5000<br>{translate_text('Services', st.session_state.language)}: {translate_text('Food, Shelter', st.session_state.language)}"
        ).add_to(m)
    st.write(translate_text("Click on markers for more details.", st.session_state.language))
    st_folium(m, width=725)


TRANSLATOR_LANGUAGES = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Arabic": "ar",
    "Italian": "it",
    "Russian": "ru",
    "Japanese": "ja",
    "Chinese (Simplified)": "zh",
    "Portuguese": "pt",
    "Hindi": "hi",
    "Chinese (Traditional)": "zh-TW"
}


def translate_page():
    st.header(translate_text("Translation Service", st.session_state.language))
    lang = st.selectbox(translate_text("Select Language", st.session_state.language), list(TRANSLATOR_LANGUAGES.keys()),
                        index=list(TRANSLATOR_LANGUAGES.values()).index(st.session_state.language))
    st.session_state.language = TRANSLATOR_LANGUAGES[lang]
    text = st.text_input(translate_text("Enter text to translate:", st.session_state.language))
    if text:
        st.write(
            f"{translate_text('Translated text: ', st.session_state.language)}: {translate_text(text, st.session_state.language)}")


def chat_page():
    st.header(translate_text("Chat with AI Assistant", st.session_state.language))
    for msg in st.session_state.history:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])

    prompt = st.text_input(translate_text("Say something", st.session_state.language))
    if prompt:
        st.session_state.history.append({'role': 'user', 'content': prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner(translate_text('💡Thinking', st.session_state.language)):
            llm, db = initialize_model_and_db()
            if llm and db:
                retriever = db.as_retriever()
                chat_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")
                response = chat_chain({"query": prompt})
                st.session_state.history.append({'role': 'Assistant', 'content': response['result']})
                with st.chat_message("Assistant"):
                    st.markdown(response['result'])


def main():
    page = option_menu(
        menu_title="Navigation",
        options=["Home", "Map", "Translate", "Chat"],
        icons=["house", "list-task", "globe", "chat"],
        orientation="horizontal",
        default_index=0
    )
    if page == "Home":
        home_page()
    elif page == "Map":
        map_page()
    elif page == "Translate":
        translate_page()
    elif page == "Chat":
        chat_page()


if __name__ == "__main__":
    main()
