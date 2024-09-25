# scripts/chat_client.py

import requests
import json

def main():
    url = "http://localhost:8000/generate"
    conversation = []
    system_message = "You are a helpful AI assistant."

    print("Welcome to the chat! Type 'quit' to exit.")
    print(f"System: {system_message}\n")

    conversation.append({"role": "system", "message": system_message})

    while True:
        user_input = input("User: ")
        if user_input.lower() == 'quit':
            break

        conversation.append({"role": "user", "message": user_input})

        payload = {
            "conversation": conversation,
            "max_length": 2048,
            "temperature": 0.7,
            "top_p": 0.9
        }

        response = requests.post(url, json=payload)

        if response.status_code == 200:
            result = response.json()
            updated_conversation = result['conversation']
            assistant_message = updated_conversation[-1]['message']
            print(f"\nAssistant: {assistant_message}\n")
            conversation = updated_conversation
        else:
            print(f"Error: {response.status_code}")
            print(response.text)

if __name__ == "__main__":
    main()