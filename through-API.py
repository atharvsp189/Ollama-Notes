import requests
import json

url = "http://localhost:11434/generate"

data = {
    "model": "apollo-deepseek",
    "prompt": "tell me a short story and make it funny."
}

response = requests.post(url, json=data, stream=False)

# check the response status
if response.status_code == 200:
    print("Generated Text:", end=" ", flush=False)
    # Iterate over the streaming response
    for line in response.iter_lines():
        if line:
            # Decode the line and parse the JSON
            decoded_line = line.decode("utf-8")
            result = json.loads(decoded_line)
            # Get the text from the response
            generated_text = result.get("response", "")
            print(generated_text, end="", flush=True)
else:
    print("Error:", response.status_code, response.text)