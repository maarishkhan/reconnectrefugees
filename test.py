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
from streamlit_extras.let_it_rain import rain
from streamlit_lottie import st_lottie
import requests
import time
import base64
from streamlit_theme import st_theme
import ollama

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
TRANSLATOR_LANGUAGES = dict(sorted({
    # We have more than 60 languages in their native tongue with the respective flag
    "English 🇺🇲": "en",
    "Español 🇪🇸": "es",
    "Français 🇲🇫": "fr",
    "Deutsch 🇩🇪": "de",
    "العربية 🇸🇦": "ar",
    "Italiano 🇮🇹": "it",
    "Русский 🇷🇺": "ru",
    "日本語 🇯🇵": "ja",
    "中文 (简体) 🇨🇳": "zh",
    "Português 🇵🇹": "pt",
    "हिन्दी🇳🇵": "hi",
    "中文 (繁體) 🇹🇼": "zh-TW",
    "தமிழ் 🇮🇳":"ta",
    "українська 🇺🇦":"uk",
    "বাংলা 🇧🇩":"bn",
    "Türkçe 🇹🇷 ":"tr",
    "한국인🇰🇷":"ko",
    "Polski 🇵🇱":"pl",
    "Tiếng Việt 🇻🇳":"vi",
    "Kiswahili 🇰🇪":"sw",
    "Nederlands 🇳🇱":"nl",
    "Yoruba 🇳🇬":"yo",
    "Tagalog 🇵🇭":"tl",
    "แบบไทย 🇹🇭":"th",
    "Urdu 🇵🇰":"ur",
    "Bahasa Indonesia 🇮🇩":"id",
    "O'zbek 🇺🇿":"uz",
    "မြန်မာ 🇲🇲":"my",
    "Hrvatski 🇭🇷":"hr",
    "አማርኛ 🇪🇹":"am",
    "नेपाली 🇳🇵":"ne",
    "Magyar 🇭🇺":"hu",
    "čeština 🇨🇿":"cs",
    "Zulu 🇿🇦":"zu",
    "Svenska 🫎":"sv",
    "български 🇧🇬":"bg",
    "հայերեն 🇦🇲":"am",
    "Shqiperia 🇦🇱":"sq",
    "Slovenský 🇸🇰":"sk",
    "Bosanski 🇧🇦":"bs",
    "ລາວ 🇱🇦":"lo",
    "Română 🇷🇴":"ro",
    "Slovenščina 🇸🇮":"sl",
    "Dansk 🇩🇰":"da",
    "Lietuvis 🇱🇹":"lt",
    "Latviski 🇱🇻":"lv",
    "Eesti keel 🇪🇪":"et",
    "Suomalainen 🇫🇮":"fi",
    "беларускі 🇧🇾":"be",
    "тоҷикӣ 🇹🇯":"tg",
    "қазақ 🇰🇿":"kk",
    "Кыргызча 🇰🇬":"ky",
    "Türkmen 🇹🇲":"tk",
    "සිංහල 🇱🇰":"si",
    "Azeri 🇦🇿":"az",
    "Norsk 🇳🇴":"no",
    "Српски 🇷🇸":"sr",
    "Melayu 🇲🇾":"ms",
    "íslenskur 🇮🇸":"is",
    "írska 🇮🇪":"ga",
    "Lëtzebuergesch 🇱🇺":"lb",
    "Malagasy 🇲🇬":"mg",
    "Somaliyeed 🇸🇴":"so",
    "ትግሪኛ 🇪🇷":"ti",
    "sesotho 🇱🇸":"st",
    "Eʋegbe 🇳🇪":"ee",
    "Lingala 🇨🇬":"ln",
    "Kinyarwanda 🇷🇼":"rw",
}.items()))


# Function to fetch city information from Teleport City API
def fetch_city_info(city_name):
    try:
        response = requests.get(f"https://api.teleport.org/api/urban_areas/slug:{city_name}/details/")
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"Error fetching city information: {e}")
        return None


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


def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = f'''
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bin_str}");
        background-size: cover;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

#set_background('photo.jpg')





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
    st.subheader(translate_text("Support", st.session_state.language))
    st.write(translate_text(
        "There are several links that you can click",
        st.session_state.language))
    st.subheader(translate_text("About Us", st.session_state.language))
    st.write(translate_text("Some cool information and pictures about us",
                            st.session_state.language))
    st.subheader(translate_text("Extra", st.session_state.language))
    st.write(translate_text("Our motivation and some extra knowledge", st.session_state.language))

def query_chatbot(query):
    try:
        messages = [
            {'role': 'system', 'content': 'You are a refugee helper with a lot of knowledge about the best places '
                                          'they should look for housing, jobs and schooling. Please answer with a brief review of the place'
                                          'ranking it out of ten and also providing an explanation of why you rate the place so'},
            {'role': 'user', 'content': str(query)},
        ]
        response = ollama.chat(model=MODEL_NAME, messages=messages)
        if response and 'message' in response and 'content' in response['message']:
            return response['message']['content']
        else:
            return "No valid response received from the chatbot."
    except Exception as e:
        st.error(f"Error querying chatbot: {e}")
        return None


def map_page():
    st.header(translate_text("Refugee Camp Locations", st.session_state.language))

    location_data = [
        {"name": "UNHCR Camp", "latitude": 40.7128, "longitude": -74.0060, "services": "Food, Shelter",
         "capacity": 5000},
        {"name": "Red Cross Center", "latitude": 34.0522, "longitude": -118.2437, "services": "Medical, Shelter",
         "capacity": 3000},
        {"name": "IRC Refugee Support", "latitude": 37.7749, "longitude": -122.4194, "services": "Food, Education",
         "capacity": 4000},
        {"name": "Local Refugee Aid", "latitude": 48.8566, "longitude": 2.3522, "services": "Shelter, Counseling",
         "capacity": 2000},
        {"name": "Global Aid Network", "latitude": 35.6895, "longitude": 139.6917, "services": "Food, Medical",
         "capacity": 6000},
        {"name": "Save the Children Camp", "latitude": 19.4326, "longitude": -99.1332, "services": "Food, Education",
         "capacity": 3500},
        {"name": "Care International Camp", "latitude": -33.8688, "longitude": 151.2093, "services": "Medical, Shelter",
         "capacity": 2500},
        {"name": "Doctors Without Borders Camp", "latitude": 51.5074, "longitude": -0.1278,
         "services": "Medical, Shelter", "capacity": 4500},
        {"name": "UNICEF Support Center", "latitude": 55.7558, "longitude": 37.6173, "services": "Food, Education",
         "capacity": 3000},
        {"name": "World Food Programme Center", "latitude": -1.2921, "longitude": 36.8219, "services": "Food, Medical",
         "capacity": 5000},
        {"name": "National Refugee Center", "latitude": 38.9072, "longitude": -77.0369,
         "services": "Shelter, Counseling", "capacity": 4000},  # Washington, D.C., USA
        {"name": "West Coast Refugee Support", "latitude": 47.6062, "longitude": -122.3321,
         "services": "Medical, Education", "capacity": 3500},  # Seattle, WA, USA
        {"name": "Southern Refugee Assistance", "latitude": 29.7604, "longitude": -95.3698, "services": "Food, Medical",
         "capacity": 3000},  # Houston, TX, USA
        {"name": "Midwest Refugee Relief", "latitude": 41.8781, "longitude": -87.6298, "services": "Shelter, Education",
         "capacity": 2800},  # Chicago, IL, USA
        {"name": "East Coast Refugee Center", "latitude": 39.9526, "longitude": -75.1652,
         "services": "Food, Counseling", "capacity": 3200},  # Philadelphia, PA, USA
        {"name": "Rocky Mountain Refugee Camp", "latitude": 39.7392, "longitude": -104.9903,
         "services": "Shelter, Medical", "capacity": 2700},  # Denver, CO, USA
        {"name": "Great Lakes Refugee Support", "latitude": 42.3314, "longitude": -83.0458, "services": "Food, Shelter",
         "capacity": 2500},  # Detroit, MI, USA
        {"name": "Northeast Refugee Assistance", "latitude": 42.3601, "longitude": -71.0589,
         "services": "Medical, Education", "capacity": 2900},  # Boston, MA, USA
        {"name": "Southwest Refugee Help", "latitude": 33.4484, "longitude": -112.0740, "services": "Food, Shelter",
         "capacity": 3100},  # Phoenix, AZ, USA
        {"name": "Florida Refugee Support", "latitude": 25.7617, "longitude": -80.1918, "services": "Medical, Shelter",
         "capacity": 3000},  # Miami, FL, USA
        {"name": "Appalachian Refugee Center", "latitude": 36.1627, "longitude": -86.7816,
         "services": "Food, Counseling", "capacity": 2000},  # Nashville, TN, USA
        {"name": "Great Plains Refugee Aid", "latitude": 39.0997, "longitude": -94.5786,
         "services": "Shelter, Education", "capacity": 2200},  # Kansas City, MO, USA
        {"name": "Pacific Northwest Refugee Help", "latitude": 45.5152, "longitude": -122.6784,
         "services": "Medical, Food", "capacity": 2800},  # Portland, OR, USA
        {"name": "Desert Refugee Center", "latitude": 36.1699, "longitude": -115.1398, "services": "Shelter, Medical",
         "capacity": 2300},  # Las Vegas, NV, USA
        {"name": "New England Refugee Assistance", "latitude": 41.8240, "longitude": -71.4128,
         "services": "Food, Shelter", "capacity": 2100},  # Providence, RI, USA
        {"name": "Mid-Atlantic Refugee Support", "latitude": 39.2904, "longitude": -76.6122,
         "services": "Shelter, Education", "capacity": 2500},  # Baltimore, MD, USA
        {"name": "Northern Refugee Aid", "latitude": 44.9778, "longitude": -93.2650, "services": "Food, Medical",
         "capacity": 2400}  # Minneapolis, MN, USA
    ]

    m = folium.Map(location=[20, 0], zoom_start=2)  # Center map around the equator for better initial view

    for camp in location_data:
        folium.Marker(
            location=[camp["latitude"], camp["longitude"]],
            popup=f"{camp['name']}<br>{translate_text('Capacity', st.session_state.language)}: {camp['capacity']}<br>{translate_text('Services', st.session_state.language)}: {translate_text(camp['services'], st.session_state.language)}"
        ).add_to(m)

    st.write(translate_text("Click on markers for more details.", st.session_state.language))
    st_folium(m, width=725)


def support_page():
    st.title("Please contribute as much as possible.", st.session_state.language)
    st.header("Volunteer", st.session_state.language)

    st.link_button("Volunteer Forever",
                   "https://www.volunteerforever.com/article_post/volunteering-abroad-for-refugee-relief/",
                   type="primary", disabled=False)

    st.link_button("International Rescue Committee",
                   "https://www.rescue.org/volunteer",
                   type="primary", disabled=False)

    st.link_button("Volunteer Headquarters",
                   "https://www.volunteerhq.org/volunteer-abroad-projects/refugee-support/",
                   type="primary", disabled=False)
    st.divider()
    st.header("Donation", st.session_state.language)

    st.link_button("Refugees International", "https://www.refugeesinternational.org/",
                   type="primary", disabled=False)

    st.link_button("UN Refugee Agency", "https://www.unhcr.org/us/", type="primary", disabled=False)

    st.link_button("Refugee Assistance Project", "https://refugeerights.org/", type="primary", disabled=False)

    st.link_button("International Association for Refugees", "https://www.iafr.org/", type="primary", disabled=False)
    st.divider()

    st.header("Language Learning", st.session_state.language)
    st.link_button("Rosetta Stone", "https://www.rosettastone.com/", type="primary", disabled=False)
    st.link_button("FluentU", "https://www.fluentu.com/", type="primary", disabled=False)
    st.link_button("Hello Talk", "https://www.hellotalk.com/?lang=en", type="primary", disabled=False)

    st.divider()
    st.header("Mental Health", st.session_state.language)
    st.link_button("SAMHSA", "https://www.samhsa.gov/find-help/national-helpline", type="primary", disabled=False)
    st.link_button("MHTTC", "https://mhttcnetwork.org/resources-support-mental-health-asylum-seekers/",
                   type="primary", disabled=False)


def chat_playground():
    st.markdown("### "+translate_text("Chat Playground", st.session_state.language))

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


def city_info_page():
    st.header(translate_text("City Information", st.session_state.language))
    city_name = st.text_input(
        translate_text("Enter city name (in slug format, e.g., 'san-francisco-bay-area'):", st.session_state.language))
    if city_name:
        city_info = fetch_city_info(city_name)
        if city_info:
            st.subheader(translate_text("City Details", st.session_state.language))
            for category in city_info['categories']:
                st.markdown(f"### {category['label']}")
                for data in category['data']:
                    st.markdown(
                        f"- **{data['label']}**: {data['value']} {data['currency_symbol'] if 'currency_symbol' in data else ''}")

def aboutUs_page():
    st.header(translate_text("About Us", st.session_state.language))
    st.image("YavuzImage.png", caption=translate_text("Yavuz Gebes", st.session_state.language),
             width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")

    st.divider()

    st.image("AarishImage.png", caption=translate_text("Mohammed Aarish Khan", st.session_state.language),
             width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")

    st.divider()

    st.image("JoshImage.png", caption=translate_text("Joshua Anand", st.session_state.language),
             width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")


def home_page():
    st.title("Being A Refugee Is Not A Choice", st.session_state.language)

    st.write("Our motivation for creating this website" +
             " was to assist incoming refugees. With an annual influx whose pr" +
             "dictability varies, the millions of refugees can either detrimentally" +
             " affect society or contribute positively in every possible aspect. Ex" +
             "amples range from economic growth through job creation and consumer s" +
             "pending to fostering innovation, creativity, and problem-solving. Failure" +
             " to address this issue will burden society with refugees, leading to inc" +
             "reased ostracism and racism. A recent report by the UNHCR highlighted t" +
             "he plight of millions of refugees facing hardships. According to the" +
             " UNHCR, as of mid-2023, the number of forcibly displaced people surpass" +
             "ed 110 million, with over 36.4 million being refugees.", st.session_state.language)

def housing_page():
    st.header("Please fill this out for your convenience.")

    with st.form("Housing related questions"):
        state_name = st.selectbox(
            "What state do you want to live in?",
            ["Alabama", "Alaska", "Arizona", "Arkansas", "California", "Connecticut", "Delaware", "Florida", "Georgia",
             "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland",
             "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada",
             "New Hampshire", "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio",
             "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", "Tennessee", "Texas", "Utah",
             "Vermont", "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming"],
            index=None,
            placeholder="Pick One"
        )

        city_name = st.text_input("What city do you want to live in?", placeholder="Enter city name")

        budget = st.slider("$Budget", min_value=250, max_value=250000, value=50000, step=100)

        renting_or_buying = st.radio("Are you open to: ", ["Renting", "Buying", "Both"], index=0, horizontal=True)

        submit_button = st.form_submit_button("Submit")

        if submit_button:
            if city_name and state_name:
                st.success("Thank you! Please wait while we pick the best location for you.")
                st.balloons()
                query = {
                    "city": city_name,
                    "state": state_name,
                    "budget": budget,
                    "preference": renting_or_buying
                }

                ranking_response = query_chatbot(query)

                if ranking_response:
                    st.write(f"Location ranked: {ranking_response}")
                else:
                    st.warning("No response received from the chatbot.")
            else:
                st.warning("Please fill in all required fields.")

# Fetch latest news
def fetch_latest_news():
    news = [
        {"title": "You can change your language below!🌎",
         "url": "https://www.unhcr.org/us/"},
        {"title": "Global News Concerning Refugees 🧬",
         "url": "https://www.aljazeera.com/"},
    ]
    return news


# Display news feed
def display_news_feed(news):
    news_feed = " | ".join(
        [f"<a href='{article['url']}' style='color: #ffffff;'>{article['title']}</a>" for article in news])
    st.markdown(f"<marquee class='news-marquee' scrollamount='11'>{news_feed}</marquee>", unsafe_allow_html=True)


def news_scroll():
    news = fetch_latest_news()
    display_news_feed(news)


def schooling_page():
    st.header("Please fill this out for your convenience.", st.session_state.language)
    reg_form = st.form("Schooling related questions", st.session_state.language)

def job_page():
    st.header("Please fill this out for your convenience.", st.session_state.language)
    reg_form = st.form("Job related questions", st.session_state.language)


def main():

    news_scroll()
    theme = st_theme()
    if theme["lightenedBg05"] == "hsla(220, 24%, 10%, 1)":
        #st.sidebar.image("darklogo.jpg", caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
        st.sidebar.markdown('<a href = "https://localhost:8501> <img src="darklogo.png"></a>',unsafe_allow_html=True)
    else:
        st.sidebar.image("logo.png", caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")

    col1, col2 = st.columns([3, 1])
    with col2:
        chat_playground()
    with col1:
        selected_lang = st.selectbox(
            "Select Language",
            list(TRANSLATOR_LANGUAGES.keys()),
            index=list(TRANSLATOR_LANGUAGES.values()).index(st.session_state.language)
        )
        if st.session_state.language != TRANSLATOR_LANGUAGES[selected_lang]:
            st.session_state.language = TRANSLATOR_LANGUAGES[selected_lang]
            rain(
                emoji="⏳",
                font_size=54,
                falling_speed=6,
                animation_length=1,
            )

        with st.sidebar:
            audio_file = open('backgroundMusic.mp3', 'rb')
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format='audio/mp3', start_time=0)
            st.write("Enjoy the background music while you use the website!", st.session_state.language)

        horizontal_menu = option_menu(
            menu_title="",
            options=[
                translate_text("Home", st.session_state.language),
                translate_text("Support", st.session_state.language),
                translate_text("Services", st.session_state.language),
                translate_text("About Us", st.session_state.language),
            ],
            icons=["house", "chat", "question-circle", "info-circle", "gear"],
            orientation="horizontal",
            styles={
                "nav-link-selected": {"background-color": "#02ab21"}
            }
        )



        if horizontal_menu == translate_text("Services", st.session_state.language):
            with st.sidebar:
                vertical_menu = option_menu(
                    menu_title="Services",
                    options=[
                        translate_text("Housing", st.session_state.language),
                        translate_text("Schooling", st.session_state.language),
                        translate_text("Job", st.session_state.language)
                    ],
                    icons=["house", "book", "briefcase", "city"],
                    orientation="vertical",
                    styles={
                        "nav-link-selected": {"background-color": "#02ab21"}
                    # set_background('backgroundImage.jpg')
                    }
                )

            if vertical_menu == translate_text("Housing", st.session_state.language):
                housing_page()
                map_page()
            elif vertical_menu == translate_text("Schooling", st.session_state.language):
                schooling_page()
                map_page()
            elif vertical_menu == translate_text("Job", st.session_state.language):
                job_page()
                map_page()

        elif horizontal_menu == translate_text("Home", st.session_state.language):
            home_page()
        elif horizontal_menu == translate_text("About Us", st.session_state.language):
            aboutUs_page()

        elif horizontal_menu == translate_text("Support", st.session_state.language):
            support_page()


# Custom CSS styling
st.markdown("""
<style>
    .css-1v3fvcr {
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 5px;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
    }
    .css-18e3th9 {
        font-size: 1.2em;
    }
    .stButton button {
        background-color: #02ab21;
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
    }
    .stButton button:hover {
        background-color: #028a18;
    }
</style>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
