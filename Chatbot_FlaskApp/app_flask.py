from flask import Flask, request, jsonify, session
from dotenv import load_dotenv
import os
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
import numpy as np
import joblib
import uuid

app = Flask(__name__)
app.secret_key = 'market-guru'

# Load environment variables
load_dotenv()

# Initialize memory length
conversational_memory_length = 100

# Load Logistic Regression model
model_file_path = 'Chatbot_FlaskApp\logistic_regression_model.pkl'
model_LR = joblib.load(model_file_path)
coef = model_LR.coef_[0]
intercept = model_LR.intercept_

# System prompt
system_prompt = f"""You are Marketing Guru, An AI-powered Marketing Assistant for Churn Prediction.\
        you help the marketing team to predict customer churn rates, \
        You only know yourself when you are asked. \
        then interactive positively offering a help, \
        if the marketing team member wants to know customer churn rate. \
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
        you can do the calculations for predicting churn rate as follows \
        Logit = {intercept} + ({coef[0]} * Age) + ({coef[1]} * Tenure) + ({coef[2]} * 'Usage Frequency) + ({coef[3]} * Support Calls) + ({coef[4]} * Payment Delay) + ({coef[5]} * Total Spend) + ({coef[6]} * Last Interaction) + ({coef[7]} * Gender_Female) + ({coef[8]} *Gender_Male) + ({coef[9]} * Subscription Type_Basic) + ({coef[10]} * 'Subscription Type_Premium) + ({coef[11]} * Subscription Type_Standard) + ({coef[12]} * Contract Length_Annual) + ({coef[13]} * Contract Length_Monthly) + ({coef[14]} * Contract Length_Quarterly)\
        churn rate = 1 / (1 + np.exp(-Logit)) \
        don not show the calculations in the chat .\
        then you can provide the churn rate to the customer and interact with him based on this result.\
        be smart and do not ask all the information at once.\
        make it a casual conversation and ask for a piece of information separately.\
        do not repeat your replies. \
        be careful in your calculations. \
"""

# Initialize Groq Langchain chat object
groq_api_key = os.getenv('GROQ_API_KEY')
if groq_api_key is None:
    raise ValueError("GROQ_API_KEY not found in environment variables.")
model = 'llama3-70b-8192'
groq_chat = ChatGroq(groq_api_key=groq_api_key, model_name=model)

# Dictionary to store conversations
conversations = {}

def create_conversation():
    memory = ConversationBufferWindowMemory(k=conversational_memory_length, 
                                            memory_key="chat_history", 
                                            return_messages=True)
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{human_input}"),
    ])
    return LLMChain(llm=groq_chat, prompt=prompt, verbose=True, memory=memory)

# Ensure each user gets a unique session ID
@app.before_request
def ensure_user_session():
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    if session['user_id'] not in conversations:
        conversations[session['user_id']] = create_conversation()

@app.route('/', methods=['POST'])
def chat():
    data = request.get_json()
    user_question = data.get('question')
    user_id = session['user_id']
    conversation = conversations[user_id]

    if not user_question:
        return jsonify({"error": "No question provided"}), 400

    # Save user question
    conversation.memory.save_context({'input': user_question}, {'output': ''})

    try:
        # Get AI response
        response = conversation.predict(human_input=user_question)
        conversation.memory.save_context({'input': ''}, {'output': response})
        return jsonify({"response": response})
    except Exception as e:
        error_message = f"Sorry, couldn't generate the answer! Error: {str(e)}"
        return jsonify({"error": error_message}), 500

@app.route('/clear', methods=['POST'])
def clear_history():
    user_id = session['user_id']
    conversations[user_id] = create_conversation()
    return jsonify({"message": "Chat history cleared"}), 200

if __name__ == "__main__":
    app.run(debug=True)
