import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from database import ai_descriptions

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
input_text = "Belows are all of the AI models that we can build.\n"
for ai in ai_descriptions:
    text = f"model name: {ai['name']}. " \
           f"model function: {ai['function']}. " \
           f"input format: {ai['input_format']}. " \
           f"output format: {ai['output_format']},\n"
    input_text += text

input_text += "We can't make an AI model that is not in this list."

doc = Document(page_content=input_text)

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents([doc])

embeddings = OpenAIEmbeddings()
db = Chroma.from_documents(texts, embeddings)

retriever = db.as_retriever()


def get_qa_chain(model_name="gpt-4o-mini"):
    """
    Create a QA chain with the specified model.
    Available models include:
    - gpt-3.5-turbo
    - gpt-3.5-turbo-16k
    - gpt-4
    - gpt-4-32k
    """
    llm = ChatOpenAI(model_name=model_name, temperature=0)

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

model_name = "gpt-4o-mini"  # You can change this to any supported model
qa = get_qa_chain(model_name)

query = "어린이집에서 아이들을 tracking 하고 그들의 가장 행복했던 순간을 캡쳐해주는 인공지능을 만들고 싶어요."
query += "\n사용가능한 모델 중 이 인공지능을 만들기 위해 필요한 모델을 알려주세요."

result = qa({"query": query})
print("Answer:", result["result"])
print("\nSource documents:")
for doc in result["source_documents"]:
    print(doc.page_content)
print()
