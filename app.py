import streamlit as st
#from chat import chat
import folium
from translate import Translator

updates_data = [
    {"title": "Update 1", "content": "First update content."},
    {"title": "Update 2", "content": "Second update content."}
]
map_data = {
    "locations": [
        {"name": "Location 1", "latitude": 40.7128, "longitude": -74.0060},
        {"name": "Location 2", "latitude": 34.0522, "longitude": -118.2437}
    ]
}
statistics_data = {
    "refugees_supported": [100, 200, 150],
    "donations_received": [5000, 7000, 6000]
}

st.set_page_config(page_title="Refugee Support", page_icon='', layout="wide", initial_sidebar_state="expanded")

languages = {
    "English": "en",
    "Español": "es",
    "Français": "fr",
    "العربية": "ar"
}

default_language = "en"

st.title("Refugee Support")
st.markdown("""
           This website provides helpful resources and information to support refugees.
           """)

st.sidebar.title("Navigation")
selected_page = st.sidebar.radio("Go to",
                                 ["Home", "Updates", "Resources", "Get Help", "Volunteer", "Donate", "News", "Chat",
                                  "Map View", "Statistics", "Forum", "Tracker", "Volunteer Opportunities",
                                  "Recommendations"])

selected_language = st.sidebar.selectbox("Select Language", list(languages.keys()),
                                         index=languages.get(default_language))
language = languages[selected_language]


def translate_text(text, lang):
    if lang != "en":
        if lang == "es":
            ts = Translator(provider="libre", from_lang="en", to_lang="es")
            translation = ts.translate(text)
            return translation
        if lang == "fr":
            ts = Translator(provider="libre", from_lang="en", to_lang="fr")
            translation = ts.translate(text)
            return translation
        if lang == "ar":
            ts = Translator(provider="libre", from_lang="en", to_lang="ar")
            translation = ts.translate(text)
            return translation


if selected_page == "Home":
    st.header(translate_text("Home", language))
    st.write(translate_text("Welcome to the Refugee Support platform.", language))


elif selected_page == "Updates":
    st.header(translate_text("Updates", language))
    for update in updates_data:
        st.subheader(update["title"])
        st.write(update["content"])

elif selected_page == "Resources":
    st.header(translate_text("Resources", lang=languages))
    categories = [
        {"title": translate_text("Legal Aid", 'en'), "description": "Find legal advice and representation.",
         "link": "https://www.example.com/legal-aid"},
        {"title": translate_text("Housing Assistance", 'en'),
         "description": "Information on finding safe and affordable housing.",
         "link": "https://www.example.com/housing-assistance"},
        {"title": translate_text("Food Banks", 'en'), "description": "Locations of local food banks and meal programs.",
         "link": "https://www.example.com/food-banks"},
        {"title": translate_text("Healthcare Services", 'en'),
         "description": "Access to medical care and health information.",
         "link": "https://www.example.com/healthcare-services"}
    ]
    for category in categories:
        st.subheader(category["title"])
        st.write(category["description"])
        st.markdown(f"[{category['title']}]({category['link']})")


elif selected_page == "Get Help":
    st.header(translate_text("Get Help", language))
    st.write(translate_text("Find ways to get involved and make a difference.", language))


elif selected_page == "Volunteer":
    st.header(translate_text("Volunteer", language))
    st.write(translate_text("Discover volunteer opportunities and apply today.", language))


elif selected_page == "Donate":
    st.header(translate_text("Donate", language))
    st.write(translate_text("Support our cause by donating today.", language))


elif selected_page == "News":
    st.header(translate_text("News", language))
    st.write(translate_text("Stay informed about the latest developments.", language))

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = ""

elif selected_page == "Chat":
    st.header(translate_text("Chat", language))
    chat_history = st.empty()
    user_input = st.text_area("Your message", height=100, value=st.session_state.chat_history)
    if st.button("Send"):
        #response = chat(user_input, language)
        response = "I don't know what to say"
        st.session_state.chat_history += f"\n**You**: {user_input}"
        st.session_state.chat_history += f"\n**Assiatant**: {response}"
        user_input = ""
        chat_history.markdown(st.session_state.chat_history)


elif selected_page == "Map View":
    st.header(translate_text("Map View", language))
    m = folium.Map(location=[37.7749, -122.4194], zoom_start=13)  # San Francisco
    for loc in map_data["locations"]:
        folium.Marker([loc["latitude"], loc["longitude"]], popup=f"{loc['name']}").add_to(m)
    st.write(m)

elif selected_page == "Statistics":
    st.header(translate_text("Statistics", language))
    st.bar_chart(statistics_data["refugees_supported"], use_container_width=True)
    st.line_chart(statistics_data["donations_received"], use_container_width=True)


elif selected_page == "Forum":
    st.header(translate_text("Forum", language))
    st.write(translate_text("Discuss, share experiences, and offer advice.", language))


elif selected_page == "Tracker":
    st.header(translate_text("Tracker", language))
    st.bar_chart(statistics_data["donations_received"], use_container_width=True)
    st.write(translate_text("Track donation amounts and categories.", language))


elif selected_page == "Volunteer Opportunities":
    st.header(translate_text("Volunteer Opportunities", language))
    st.write(translate_text("Post volunteering opportunities and manage applications.", language))


elif selected_page == "Recommendations":
    st.header(translate_text("Recommendations", language))
    st.write(translate_text("Based on your preferences and activity, we recommend resources, news, and ways to help.",
                            language))
