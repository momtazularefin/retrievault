import os
import sys
from langchain_openai import ChatOpenAI
from retrievault.config import get_settings

settings = get_settings()
key = settings.openai_api_key.strip()
if not key:
    print("Error: OPENAI_API_KEY is empty in configuration settings.")
    sys.exit(1)

print(f"Testing OpenAI with model: gpt-4o-mini")
print(f"API Key prefix: {key[:15]}...")

llm = ChatOpenAI(model="gpt-4o-mini", api_key=key, timeout=15)
try:
    print("Sending request...")
    resp = llm.invoke("Say 'Hello OpenAI'")
    print("Success! Response:", resp.content)
except Exception as e:
    print("Connection failed!")
    print(f"Error Type: {type(e)}")
    print(f"Error Message: {e}")
