from langchain_groq import ChatGroq

import getpass
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
    print("loaded .env file")
except ModuleNotFoundError:
    pass

def _set_if_undefined(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"Please provide your {var}: ")


_set_if_undefined("GROQ_API_KEY")
# _set_if_undefined("NEON_API_KEY")
model = ChatGroq(model="openai/gpt-oss-120b")
