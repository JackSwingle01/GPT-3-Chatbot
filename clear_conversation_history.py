# this script deletes the contents of the text_logs and vector_logs folders

import os

def clear_text_logs():
    files = os.listdir("text_logs")
    for file in files:
        os.remove(f"text_logs/{file}")

def clear_vector_logs():
    files = os.listdir("vector_logs")
    for file in files:
        os.remove(f"vector_logs/{file}")

if __name__ == "__main__":
    clear_text_logs()
    clear_vector_logs()
    