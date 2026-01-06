# def main():
#     # import langchainhub
#     # print(langchainhub.__file__)

#     # print("Hello from agentic-rag!\n\n")

#     # from langchainhub import Client

#     # client = Client()
#     # prompt = client.pull("rlm/rag-prompt")

#     # print(prompt)

#     from graph.graph import app

#     print("Hello from agentic-rag!\n\n")
#     result = app.invoke({"question": "What is agent memory?"})
#     print(result)


# if __name__ == "__main__":
#     main()

# main.py
import os
from dotenv import load_dotenv

load_dotenv()


def main():
    from graph.graph import app

    inputs = {"question": "What is agent memory?"}

    for output in app.stream(inputs):
        for key, value in output.items():
            print(f"Node '{key}':")
            print(value)
        print("\n---\n")


if __name__ == "__main__":
    main()
