import requests
import openai
import os
import json

client = openai.OpenAI(
    api_key = os.getenv("POE_API_KEY"),
    base_url = "https://api.poe.com/v1"
)

def get_weather(latitude, longitude):
    response = requests.get(f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m")
    data = response.json()
    return data['current']['temperature_2m']

tools = [{
    "type": "function",
    "name": "get_weather",
    "description": "Get current temperature for provided coordinates in celsius.",
    "parameters": {
        "type": "object",
        "properties": {
            "latitude": {"type": "number"},
            "longitude": {"type": "number"}
        },
        "required": ["latitude", "longitude"],
        "additionalProperties": False
    },
    "strict": True
}]

input_messages = [{"role": "user", "content": "What's the weather like in Paris today?"}]

response = client.responses.create(
    model="gpt-5-chat",
    input=input_messages,
    tools=tools,
)

# Extract the tool call and arguments
tool_call = response.output[0]
args = json.loads(tool_call.arguments)
# Call the function
result = get_weather(args["latitude"], args["longitude"])

# Append the tool call and result to the input messages
input_messages.append(tool_call) # append model's function call message
input_messages.append({ # append result message
    "type": "function_call_output",
    "call_id": tool_call.call_id,
    "output": str(result)
})

response_2 = client.responses.create(
    model="gpt-5-chat",
    input=input_messages,
    tools=tools,
)
print(response_2.output_text)