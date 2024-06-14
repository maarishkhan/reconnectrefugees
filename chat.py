import os
from langchain.embeddings import OllamaEmbeddings
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


class SimpleEmbeddingStore:
    def __init__(self):
        self.data = {}
        self.ids = []

    def add_embedding(self, key, embedding):
        self.data[key] = embedding
        self.ids.append(key)

    def get_embedding(self, key):
        return self.data.get(key)

    def as_retriever(self):
        return [(id_, self.get_embedding(id_)) for id_ in self.ids]


class ConversationManager:
    def __init__(self):
        self.conversations = []
        self.text_data = []
        self.embedding_store = SimpleEmbeddingStore()

    def add_message(self, role, content):
        self.conversations.append({"role": role, "content": content})

    def get_conversations(self):
        return self.conversations

    def load_txt_data(self, txtDir, embed):
        for filename in os.listdir(txtDir):
            if filename.endswith('.txt'):
                filepath = os.path.join(txtDir, filename)
                with open(filepath, 'r', encoding='utf-8') as file:
                    text = file.read()
                    embedding = embed.embed(text)
                    self.embedding_store.add_embedding(filename, embedding)
                    self.text_data.append(text)


llm = Ollama(model="tinyllama", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
embeddings = OllamaEmbeddings()

conversation_manager = ConversationManager()
txt_dir = "data"

conversation_manager.load_txt_data(txt_dir, embeddings)


retriever = conversation_manager.embedding_store.as_retriever()


query_chain = RetrievalQA(retriever=retriever)


def chat(prompt, language):
    conversation_manager.add_message("user", prompt)
    context = " ".join(conversation_manager.text_data)
    filled_prompt = (f"As a Humanitarian Aid Worker, your primary goal is to assist refugees by addressing their "
                     f"questions promptly and clearly. Regardless of the language in which the queries are posed, "
                     f"strive to respond in the same language for effective communication and understanding. Offer "
                     f"concise and professional answers to ensure the refugees receive the support they need in a "
                     f"timely manner. Remember, your responses should be both informative and empathetic to meet the "
                     f"refugees' diverse needs and situations. You should answer in {language}\n\nContext: {context}\n"
                     f"Question: {prompt}\nAnswer:")
    response = query_chain({"query": filled_prompt})
    reply = response['result']
    conversation_manager.add_message("assistant", reply)
    return reply
