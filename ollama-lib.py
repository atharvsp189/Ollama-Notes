import ollama


response = ollama.list()

print(response)

# == Chat example ==
res = ollama.chat(
    model="deepseek-r1",
    messages=[
        {"role": "user", "content": "why is the sky blue?"},
    ],
)
print(res["message"]["content"])


res = ollama.chat(
    model="llama3.2",
    messages=[
        {
            "role": "user",
            "content": "why is the ocean so salty?",
        },
    ],
    stream=False,
)


res = ollama.generate(
    model="llama3.2",
    prompt="why is the sky blue?",
)

# show
print(ollama.show("llama3.2"))


# Create a new model with modelfile
modelfile = """
FROM llama3.2
SYSTEM You are very smart assistant who knows everything about oceans. You are very succinct and informative.
PARAMETER temperature 0.1
"""

ollama.create(model="knowitall", modelfile=modelfile)

res = ollama.generate(model="knowitall", prompt="why is the ocean so salty?")
print(res["response"])


# delete model
ollama.delete("knowitall")