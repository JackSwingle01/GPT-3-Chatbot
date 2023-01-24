import openai
import os
from time import time
from uuid import uuid4
import json
import numpy as np


def write_file(filename, text):
    with open(filename, 'w', encoding="utf-8") as f:
        return f.write(text)


def read_file(filename):
    with open(filename, 'r', encoding="utf-8") as f:
        return f.read()


# saves a json file
def write_json_file(filename, data):
    with open(filename, 'w', encoding="utf-8") as f:
        return json.dump(data, f, indent=4, ensure_ascii=False, sort_keys=True)


# loads a json file
def read_json_file(filename):
    with open(filename, 'r', encoding="utf-8") as f:
        return json.load(f)


# makes api call to openai and returns the completion
# saves the prompt and response to a file
def get_completion(prompt, model="text-davinci-003", temperature=.7, max_tokens=500, stop=None):

    response = openai.Completion.create(
        model=model,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        stop=stop
    )
    text = response["choices"][0]["text"].strip()
    # Todo: remove \t and \r\n from text
    filename = f"{time()}_log.txt"
    if not os.path.exists("text_logs"):
        os.makedirs("text_logs")
    write_file(f"text_logs/{filename}", prompt + "\n\n----------\n\n" + text)
    return text


def vector_embedding(text, model="text-embedding-ada-002"):
    response = openai.Embedding.create(input=text, model=model)
    vector = response["data"][0]["embedding"]
    return vector

# gets all the previous conversation vectors and returns them as a list sorted chronologically


def load_conversation():
    files = os.listdir("vector_logs")
    result = list()
    for file in files:
        result.append(read_json_file(f"vector_logs/{file}"))
    return sorted(result, key=lambda i: i['time'], reverse=False)


def cosine_similarity(vector1, vector2):
    return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))


def fetch_memories(vector, logs, n):
    scores = list()
    for log in logs:
        if vector == log['vector']:
            continue
        score = cosine_similarity(vector, log['vector'])
        log['score'] = score
        scores.append(log)
    scores = sorted(scores, key=lambda i: i['score'], reverse=True)
    try:
        return scores[0:n]
    except:
        return scores


def summarize_memories(memories):
    memories = sorted(memories, key=lambda i: i['time'], reverse=False)
    block = ""
    for memory in memories:
        block += f"{memory['speaker']}: {memory['text']}\n\n"
    block = block.strip()
    prompt = read_file("prompt_notes.txt").replace("<<INPUT>>", block)
    notes = get_completion(prompt, model="text-curie-001", temperature=0)
    return notes.strip()


def get_recent_messages(conversation, n):
    try:
        recent_messages = conversation[-n:]
    except:
        recent_messages = conversation
    output = ""
    for message in recent_messages:
        output += f"{message['speaker']}: {message['text']}\n\n"
    output = output.strip()
    return output


if __name__ == "__main__":

    openai.api_key = read_file("API_KEY_PRIVATE.txt")

    while True:
        # Get user input, vectorize it, and save it to a file as json
        user_input = input("USER:")
        vector = vector_embedding(user_input)
        info = {"speaker": "USER", "time": time(), "vector": vector,
                "text": user_input, "uuid": str(uuid4())}
        filename = f"log_{time()}_USER.txt"
        write_json_file(f"vector_logs/{filename}", info)

        # Load conversation
        conversation = load_conversation()
        # compose prompt
        if len(conversation) > 1:
            memories = fetch_memories(vector, conversation, 10)
            # TODO - fetch declarative memories (facts, wikis, KB, company data, internet, etc)
            notes = summarize_memories(memories)
            recent = get_recent_messages(conversation, 4)
            prompt = read_file("prompt_response.txt").replace(
                "<<NOTES>>", notes).replace("<<RECENT>>", recent)
        else:
            prompt = read_file("prompt_response.txt").replace(
                "<<NOTES>>", "*This is the first message, there are no notes yet.*").replace("<<RECENT>>", "*This is the first message, there are no recent messages yet.*")

        # generate response, vectorize it, and save it to a file as json
        output = get_completion(prompt, temperature=.7 ,stop=["USER:", "ATHENA:"])
        vector = vector_embedding(output)
        info = {"speaker": "ATHENA", "time": time(), "vector": vector,
                "text": output, "uuid": str(uuid4())}
        filename = f"log_{time()}_ATHENA.txt"
        write_json_file(f"vector_logs/{filename}", info)

        # print output
        print(f"\n\nBOT: \n{output}\n\n")
