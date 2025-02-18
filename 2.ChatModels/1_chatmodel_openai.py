from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4") # currently 4o mini is the cheapest model


result = model.invoke("what is the capital of nepal?")

print(result)

print("Content => " + result.content)