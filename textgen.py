import os
import dotenv
import numpy as np
from google import genai

dotenv.load_dotenv()

def generate_summary(text):
    client = genai.Client(api_key=os.environ['API_KEY'])
    response = client.models.generate_content(
        model="gemini-2.0-flash", 
        contents=(f"Summarize this text in 5 sentences: ", text)
    )
    return response.text

def generate_response(text, instructions = "You are a helpful assistant.", model_name = "gemini-2.0-flash"):
    client = genai.Client(api_key=os.environ['API_KEY'])
    prompt = f"Summarize this text in 5 sentences, and only return the summary: {text}"
    full_prompt = f"instructions: {instructions}\n\nprompt: {prompt}"
    try:
        response = client.models.generate_content(
            model=model_name, 
            contents=full_prompt
        )
        return response.text
    except Exception as e:
        print(f"error: {e}")
        return "Gemini couldn't handle your prompt"

if __name__ == "__main__":
    print("type quit to exit")
    while True:
        user_prompt = input("you: ")
        if user_prompt.strip().lower() == "quit":
            print("stopped")
            break
        print(f"Gemini: {generate_response(user_prompt)}")
    

#Add more function if necessary