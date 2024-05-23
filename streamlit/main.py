import json

import requests

import streamlit as st
from streamlit.logger import get_logger

model_api_url = "http://model-api:8080/predictions/my_tc"


# Streamed response emulator
def response_generator(prompt):
    obj = {"text": prompt, "target": 1}
    return requests.post(model_api_url, data=json.dumps(obj)).text


if __name__ == "__main__":
    logger = get_logger(__name__)
    st.title("Hyakumanben bot")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What is up?"):
        # Add user message to chat history
        logger.info(f"User Input: {prompt}")
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            response = response_generator(prompt)
            logger.info(f"Assistant Output: {response}")
            st.write(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
