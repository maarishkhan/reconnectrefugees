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
from openai import OpenAI

load_dotenv()

embeddings = OllamaEmbeddings()

if "history" not in st.session_state:
    st.session_state.history = []

if "language" not in st.session_state:
    st.session_state.language = "en"

st.set_page_config(
    page_title="Refugee Assistance Hub",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_NAME = "tinyllama:latest"
PERSIST_DIRECTORY = "chromadb"
DATA_DIRECTORY = "./data/"
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
    for camp, cords in location_data.items():
        folium.Marker(
            cords,
            popup=f"{camp}<br>{translate_text('Capacity', st.session_state.language)}: "
                  f"5000<br>{translate_text('Services', st.session_state.language)}: "
                  f"{translate_text('Food, Shelter', st.session_state.language)}"
        ).add_to(m)
    st.write(translate_text("Click on markers for more details.", st.session_state.language))
    st_folium(m, width=725)


def chat_playground():
    st.header(translate_text("Chat Playground", st.session_state.language))

    client = OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
    )

    message_container = st.container()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        avatar = "🤖" if message["role"] == "assistant" else "😎"
        with message_container.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    if prompt := st.chat_input(translate_text("Enter a prompt here...", st.session_state.language)):
        try:
            st.session_state.messages.append(
                {"role": "user", "content": prompt})

            message_container.chat_message("user", avatar="😎").markdown(prompt)

            with message_container.chat_message("assistant", avatar="🤖"):
                with st.spinner(translate_text("Model working...", st.session_state.language)):
                    stream = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[
                            {"role": m["role"], "content": m["content"]}
                            for m in st.session_state.messages
                        ],
                        stream=True,
                    )
                # stream response
                response = st.write_stream(stream)
            st.session_state.messages.append(
                {"role": "assistant", "content": response})

        except Exception as e:
            st.error(e, icon="⛔️")


def main():
    selected_lang = st.sidebar.selectbox(
        translate_text("Select Language", st.session_state.language),
        list(TRANSLATOR_LANGUAGES.keys()),
        index=list(TRANSLATOR_LANGUAGES.values()).index(st.session_state.language)
    )
    st.session_state.language = TRANSLATOR_LANGUAGES[selected_lang]

    # Navigation Menu
    page = option_menu(
        menu_title=translate_text("Navigation", st.session_state.language),
        options=[translate_text("Home", st.session_state.language),
                 translate_text("Map", st.session_state.language),
                 translate_text("Chat Playground", st.session_state.language)],
        icons=["house", "map", "chat"],
        orientation="horizontal",
        default_index=0
    )

    if page == translate_text("Home", st.session_state.language):
        home_page()
    elif page == translate_text("Map", st.session_state.language):
        map_page()
    elif page == translate_text("Chat Playground", st.session_state.language):
        chat_playground()


if __name__ == "__main__":
    main()
