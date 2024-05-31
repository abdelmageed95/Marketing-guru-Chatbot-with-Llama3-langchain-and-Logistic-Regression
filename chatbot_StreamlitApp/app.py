import streamlit as st
from dotenv import load_dotenv
import os
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

def main():
    #st.set_page_config(page_title="PandasAI", page_icon="🐼")
    st.title("Marketing Guru 📊 ")
    st.subheader("An AI-Powered Marketing Assistant for Churn Prediction")

    # Side Menu Bar
    with st.sidebar:
        #st.title("Configuration:⚙️")
        #st.text("Data Setup: 📝")
        #st.markdown(":green[*Please ensure the first row has the column names.*]")
        
        #conversational_memory_length = st.slider('Conversational memory length:', 1, 100, value=5)
        st.text("Click to Clear Chat history")
        if st.button("CLEAR 🗑️"):
            st.session_state.chat_history = []

    # Get Groq API key
    groq_api_key = os.environ.get('GROQ_API_KEY')
    if not groq_api_key:
        st.error("GROQ_API_KEY not found in environment variables.")
        return
    
    conversational_memory_length = 100
    
    system_prompt = """You are Marketing Guru, An AI-powered Marketing Assistant for Churn Prediction.\
        you help the marketing team to predict customer churn rates, \
        You only know yourself when you are asked. \
        then interactive positively offering a help, \
        if the marketing team member wants to know customer churn rates, \
        you will collect some information from him to make the calculation for predicting churn rate.\
        you will collect the next information :\
        Age of the customer, accept integer or float\
        Gender of the customer , only accept male or female. \
        Tenure: How long the customer been with your service/ company on average? accept integer or float\
        Usage Frequency: How frequently do customers use your service, on average? accept integer or float \
        Support Calls: What is the average number of support calls made by customers? accept integer or float \
        Payment Delay: how many times the customer delays the payment, accept integer or float \
        Subscription Type: is the Subscription Basic, Standard, or Premium, Put the choosen type will = 1 and the other types = 0 \
        Contract Length: is the contract Monthly, Quarterly, or Annual, Put the chosen type will = 1 and the other types = 0 \
        Total Spend: What is the average total spend of customers, accept integer or float \
        Last Interaction: How recently did customers interact with your service? accept integer or float\
        then construct this a dictionary like this example \
        example_data = {
                    'Age': 30,
                    'Tenure': 39, 
                    'Usage Frequency': 14, 
                    'Support Calls': 5, 
                    'Payment Delay': 18,
                    'Total Spend': 932, 
                    'Last Interaction': 17, 
                    'Gender_Female': 1, 
                    'Gender_Male': 0,
                    'Subscription Type_Basic': 0, 
                    'Subscription Type_Premium': 0,
                    'Subscription Type_Standard': 1, 
                    'Contract Length_Annual': 1,
                    'Contract Length_Monthly': 0, 
                    'Contract Length_Quarterly': 0
                }

        once you get all the information for the dictionary, you can do the calculations for predicting churn rate as follows \
        Logit = 0.9499 + (0.0356 * example_data['Age']) + (-0.0078 * example_data['Tenure']) + (-0.0145 * example_data['Usage Frequency']) + (0.7476 * example_data['Support Calls']) + (0.1126 * example_data['Payment Delay']) + (-0.0060 * example_data['Total Spend']) + (0.0607 * example_data['Last Interaction']) + (1.0499 * example_data['Gender_Female']) + (-0.1001 * example_data['Gender_Male']) + (0.3954 * example_data['Subscription Type_Basic']) + (0.2757 * example_data['Subscription Type_Premium']) + (0.2787 * example_data['Subscription Type_Standard']) + (-3.1907 * example_data['Contract Length_Annual']) + (7.3286 * example_data['Contract Length_Monthly']) + (-3.1880 * example_data['Contract Length_Quarterly'])\
        churn rate = 1 / (1 + np.exp(-Logit)) \
        then you can provide the churn rate to the customer and interact with him based on this result.\
        be smart and do not ask all the information at once.\
        make it a causal conversation and ask for a piece of information separately.\
        do not repeat your replies. \
        be careful in your calculations. \
        """
    
    model = 'llama3-70b-8192'

    memory = ConversationBufferWindowMemory(k=conversational_memory_length, 
                                            memory_key="chat_history", 
                                            return_messages=True)

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    else:
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                memory.save_context({'input': message['question']}, {'output': ''})
            elif message['role'] == 'assistant':
                memory.save_context({'input': ''}, {'output': message['response']})

    groq_chat = ChatGroq(groq_api_key=groq_api_key, model_name=model)

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{human_input}"),
        ]
    )

    conversation = LLMChain(
        llm=groq_chat,
        prompt=prompt,
        verbose=True,
        memory=memory,
    )

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            if 'question' in message:
                st.markdown(message["question"])
            elif 'response' in message:
                st.write(message["response"])
            elif 'error' in message:
                st.text(message['error'])

    user_question = st.chat_input("... ")

    if user_question:
        with st.chat_message("user"):
            st.markdown(user_question)
        st.session_state.chat_history.append({"role": "user", "question": user_question})

        try:
            with st.spinner("..."):
                response = conversation.predict(human_input=user_question)
                with st.chat_message("assistant"):
                    st.write(response)
                st.session_state.chat_history.append({"role": "assistant", "response": response})
        except Exception as e:
            error_message = f"⚠️ Sorry, couldn't generate the answer! Please try rephrasing your question. Error: {str(e)}"
            with st.chat_message("assistant"):
                st.text(error_message)
            st.session_state.chat_history.append({"role": "assistant", "error": error_message})

if __name__ == "__main__":
    main()
