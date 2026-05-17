import os
from dotenv                                         import load_dotenv
from langchain_core.prompts                         import ChatPromptTemplate
from langchain_core.messages                        import HumanMessage
from langchain_openai                               import ChatOpenAI,OpenAIEmbeddings

load_dotenv()

def main():
    print("Hello from financebench-rag-eval!")


if __name__ == "__main__":
    main()
