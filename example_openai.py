from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()


response = client.responses.create(
    model="gpt-4o-mini",
    tools=[{"type": "web_search_preview"}],
    input="What are the top5 global news story for today?"
)

print(response.output_text)