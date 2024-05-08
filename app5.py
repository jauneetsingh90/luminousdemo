import os
from pathlib import Path
import boto3
from langchain_openai import ChatOpenAI
from langchain_community.chat_models.bedrock import BedrockChat
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.astradb import AstraDB
from langchain.prompts.chat import ChatPromptTemplate
from langchain.memory import ConversationBufferWindowMemory, AstraDBChatMessageHistory
from langchain.schema import HumanMessage, AIMessage
from langchain.schema.runnable import RunnableMap
from langchain.callbacks.base import BaseCallbackHandler

import streamlit as st

from dotenv import load_dotenv
load_dotenv()
st.set_page_config(initial_sidebar_state="collapsed")

# Load the environment variables,from either Streamlit Secrets or .env file
# After loading then insert into environment using os.environ
# Keyspace and Collection are set here in the code.
# If using LangChain, set LANGCHAIN_TRACING_V2 to 'true'

LOCAL_SECRETS = False

# If running locally, then use .env file, or use local Streamlit Secrets
if LOCAL_SECRETS:
    ASTRA_DB_APPLICATION_TOKEN = os.environ["ASTRA_DB_APPLICATION_TOKEN"]
    ASTRA_VECTOR_ENDPOINT = os.environ["ASTRA_VECTOR_ENDPOINT"]
    OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
    AWS_ACCESS_KEY_ID = os.environ["AWS_ACCESS_KEY_ID"]
    AWS_SECRET_ACCESS_KEY = os.environ["AWS_SECRET_ACCESS_KEY"]
    AWS_DEFAULT_REGION = os.environ["AWS_DEFAULT_REGION"]

# If running in Streamlit, then use Streamlit Secrets
else:
    ASTRA_DB_APPLICATION_TOKEN = st.secrets["ASTRA_DB_APPLICATION_TOKEN"]
    ASTRA_VECTOR_ENDPOINT = st.secrets["ASTRA_VECTOR_ENDPOINT"]
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    AWS_ACCESS_KEY_ID = st.secrets["AWS_ACCESS_KEY_ID"]
    AWS_SECRET_ACCESS_KEY = st.secrets["AWS_SECRET_ACCESS_KEY"]
    AWS_DEFAULT_REGION = st.secrets["AWS_DEFAULT_REGION"]


os.environ["AWS_ACCESS_KEY_ID"] = AWS_ACCESS_KEY_ID
os.environ["AWS_SECRET_ACCESS_KEY"] = AWS_SECRET_ACCESS_KEY
os.environ["AWS_DEFAULT_REGION"] = AWS_DEFAULT_REGION

ASTRA_DB_KEYSPACE = "default_keyspace"
ASTRA_DB_COLLECTION = "luminous"

os.environ["LANGCHAIN_PROJECT"] = "blueillusion"
os.environ["LANGCHAIN_TRACING_V2"] = "true"


print("Started")


# Streaming call back handler for responses
import re



class StreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs):
        # Continuously update the text with each new token
        self.text += token
        # Display updated text with cursor to indicate ongoing typing
        self.container.markdown(self.text + "▌", unsafe_allow_html=True)

    def on_llm_end_of_transmission(self):
        # Remove typing cursor by updating with final text
        self.container.markdown(self.text, unsafe_allow_html=True)
        # Process and append a clickable product link banner if applicable
        self.append_product_link_banner(self.text)

    def append_product_link_banner(self, text):
        product_name = self.extract_product_name(text)
        if product_name:
            banner_html = self.create_product_link_banner(product_name)
            self.container.markdown(banner_html, unsafe_allow_html=True)

    @staticmethod
    def extract_product_name(text):
        match = re.search(r"Buy Now:\s*(.*)", text)
        if match:
            return match.group(1).strip()
        return None

    @staticmethod
    def create_product_link_banner(product_name):
        # Construct a URL-safe product name for the link
        product_url = f"https://www.yourwebsite.com/product/{product_name.replace(' ', '-').lower()}"
        return f"""
        <a href="{product_url}" target="_blank" style="text-decoration: none;">
            <div style="background-color: #0043af; color: white; padding: 10px; text-align: center;">
                Buy Now: {product_name}
            </div>
        </a>
        """


#################
### Constants ###
#################

# Define the number of docs to retrieve from the vectorstore and memory
top_k_vectorstore = 8
top_k_memory = 3

###############
### Globals ###
###############

global lang_dict
global rails_dict
global embedding
global vectorstore
global retriever
global model
global chat_history
global memory


#############
### Login ###
#############
# Close off the app using a password




def check_username():
    """Prompts for and validates the username interactively within the chatbot."""
    greeting_message = "Welcome to Luminous India, how can I assist you?"
    username_prompt = "Please enter your username to continue:"

    # Display the greeting message only initially
    if 'username_valid' not in st.session_state:
        st.write(greeting_message)
    
    # Request the username from the user
    username = st.text_input(username_prompt, key='username')
    
    # Validate the username once input is provided
    if username:  # Checks if any username has been entered
        st.session_state['username_valid'] = True
        st.session_state.user = username  # Set the username in the session state
    else:
        st.session_state['username_valid'] = False

    return st.session_state.get('username_valid', False)

def logout():
    keys_to_delete = ['username_valid', 'user', 'messages']
    for key in keys_to_delete:
        if key in st.session_state:
            del st.session_state[key]
    # Reset functions to clear chat history and memory if they exist
    if 'load_chat_history' in globals():
        load_chat_history.clear()
    if 'load_memory' in globals():
        load_memory.clear()
    if 'load_retriever' in globals():
        load_retriever.clear()

# Initiate username check and validation
if not check_username():
    st.stop()  # Stop execution if a username is not provided.

username = st.session_state.user  # Access the username






#######################
### Resources Cache ###
#######################

# Cache boto3 session for future runs
@st.cache_resource(show_spinner='Getting the Boto Session...')
def load_boto_client():
    print("load_boto_client")
    session = boto3.Session(
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_DEFAULT_REGION
    )
    return boto3.client("bedrock-runtime")

# Cache OpenAI Embedding for future runs
@st.cache_resource(show_spinner='Getting the Embedding Model...')
def load_embedding():
    print("load_embedding")
    # Get the OpenAI Embedding
    return OpenAIEmbeddings()
    # Bedrock Option - if we want use Bedrock fro emebeddings
    # Get the Bedrock Embedding
    #return BedrockEmbeddings(credentials_profile_name="default", region_name="us-east-1")

    

# Cache Vector Store for future runs
@st.cache_resource(show_spinner='Getting the Vector Store from Astra DB...')
def load_vectorstore():
    print(f"load_vectorstore: {ASTRA_DB_KEYSPACE} / {ASTRA_DB_COLLECTION}")
    # Get the load_vectorstore store from Astra DB
    return AstraDB(
        embedding=embedding,
        namespace=ASTRA_DB_KEYSPACE,
        collection_name=ASTRA_DB_COLLECTION,
        token=ASTRA_DB_APPLICATION_TOKEN,
        api_endpoint=ASTRA_VECTOR_ENDPOINT,
    )
    
# Cache Retriever for future runs
@st.cache_resource(show_spinner='Getting the retriever...')
def load_retriever():
    print("load_retriever")
    # Get the Retriever from the Vectorstore
    return vectorstore.as_retriever(
        search_kwargs={"k": top_k_vectorstore}
    )

# Cache Chat Model for future runs
@st.cache_resource(show_spinner='Getting the Chat Model...')
def load_model(model_id="openai.gpt-3.5"):
    print(f"load_model: {model_id}")
    # if model_id contains 'openai' then use OpenAI model
    if 'openai' in model_id:
        if '3.5' in model_id:
            gpt_version = 'gpt-3.5-turbo'
        else:
            gpt_version = 'gpt-4-turbo-preview'
        return ChatOpenAI(
            temperature=0.2,
            model=gpt_version,
            streaming=True,
            verbose=True
            )
    # else use Bedrock model
    return BedrockChat(
        #credentials_profile_name="default",  # we are using the boto3 client instead
        #region_name="us-east-1",             # we are using the boto3 client instead
        client=bedrock_runtime,
        model_id=model_id,
        streaming=True,
        model_kwargs={"temperature": 0.2},
    )

# Cache Chat History for future runs
@st.cache_resource(show_spinner='Getting the Message History from Astra DB...')
def load_chat_history(username):
    print("load_chat_history")
    return AstraDBChatMessageHistory(
        session_id=username,
        token=ASTRA_DB_APPLICATION_TOKEN,
        api_endpoint=ASTRA_VECTOR_ENDPOINT,
        namespace=ASTRA_DB_KEYSPACE,
    )

@st.cache_resource(show_spinner='Getting the Message History from Astra DB...')
def load_memory():
    print("load_memory")
    return ConversationBufferWindowMemory(
        chat_memory=chat_history,
        return_messages=True,
        k=top_k_memory,
        memory_key="chat_history",
        input_key="question",
        output_key='answer',
    )

# Cache prompt
# 
#Do not include images in your response.
#Provide at most 2 items that are relevant to the user's question.
#You're friendly and you answer extensively with multiple sentences.
#You prefer to use bulletpoints to summarize.
#Focus on the user's needs and provide the best possible answer.

@st.cache_data()
def load_prompt():
    print("load_prompt")
    template = """You're a helpful  assistant tasked to help users for buying invertors and other electrical products for households and other places 
You like to help a user find the perfect outfit for a special occasion.
Given the pricing 
You should also suggest the batteries which are best for the invertors
Prompt the user with clarifying questions so that you know at least for what they need an invertor for example recommend invertor for 2 BHK, 3 BHK. Please give multiple option for example if two invertors below 10K satisfy requirement then give 2 results
Do not include any information other than what is provided in the context below.
Include an  image of the product taken from the image attribute in the metadata. and also display the image, if image is not there open the link and show it as banner
Include the price of the product if found in the context.
Include a link to buy each item you recommend if found in the context. Here is a sample buy link:
Include a link from the link metadata field, please open the link and show that as banner
Also if somebody says Hi, hey or any other greeting, please response how i can help you
If you don't know the answer, just say 'I do not know the answer'.
If the user has not asked a question related to Luminous Products, you can respond with 'visit Luminous India websites https://www.luminousindia.com/'.

Use the following context to answer the question:
{context}

Use the previous chat history to answer the question:
{chat_history}

Question:
{question}

Answer in English"""

    return ChatPromptTemplate.from_messages([("system", template)])



#####################
### Session state ###
#####################

# Start with empty messages, stored in session state
 #Draw all messages, both user and agent so far (every time the app reruns)
 
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize
bedrock_runtime = load_boto_client()
embedding = load_embedding()
vectorstore = load_vectorstore()
retriever = load_retriever()
chat_history = load_chat_history(username)
memory = load_memory()
prompt = load_prompt()

model_id = 'openai.gpt-3.5'  # Change this to the desired model ID
model = load_model(model_id)


# Draw all messages, both user and agent so far (every time the app reruns)
if st.session_state.messages:  # Check if there are any messages to display
    for message in st.session_state.messages:
        st.chat_message(message.type).markdown(message.content)

# Now get a prompt from a user
if question := st.chat_input("How can I help you?"):
    print(f"Got question: \"{question}\"")

    # Add the prompt to messages, stored in session state
    st.session_state.messages.append(HumanMessage(content=question))

    # Draw the prompt on the page
    print("Display user prompt")
    with st.chat_message("user"):
        st.markdown(question)

    # Get the results from Langchain
    print("Get AI response")
    with st.chat_message("assistant"):
        # UI placeholder to start filling with agent response
        response_placeholder = st.empty()

        history = memory.load_memory_variables({})
        print(f"Using memory: {history}")

        inputs = RunnableMap({
            'context': lambda x: retriever.get_relevant_documents(x['question']),
            'chat_history': lambda x: x['chat_history'],
            'question': lambda x: x['question']
        })
        print(f"Using inputs: {inputs}")

        chain = inputs | prompt | model
        print(f"Using chain: {chain}")

        # Call the chain and stream the results into the UI
        response = chain.invoke({'question': question, 'chat_history': history}, config={'callbacks': [StreamHandler(response_placeholder)], "tags": [username]})
        print(f"Response: {response}")
        #print(embedding.embed_query(question))
        content = response.content

        # Write the final answer without the cursor
        response_placeholder.markdown(content)


        # Add the result to memory
        memory.save_context({'question': question}, {'answer': content})

        # Add the answer to the messages session state
        st.session_state.messages.append(AIMessage(content=content))