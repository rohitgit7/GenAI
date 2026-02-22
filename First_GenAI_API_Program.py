import openai
import os

client = openai.OpenAI(
    api_key = os.getenv("POE_API_KEY"),
    base_url = "https://api.poe.com/v1"
)

chat = client.chat.completions.create(
    model = "gpt-5-chat",
    messages = [{
      "role": "user",
      "content": "can I login to claude desktop using poe credentials?"
    }]

)

print(chat.choices[0].message.content)