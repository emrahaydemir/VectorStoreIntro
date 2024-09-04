from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from typing import List
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.prompts import ChatMessagePromptTemplate


load_dotenv()


documents = [  # vector documentleri olustur. bunlar dosyalar da olabilir, metadata istege baglidir. fakat source belirteci daha anlasilabilir olmasi acisindan onemli
    Document(
        page_content="Dogs are great companions, known for their loyalty and friendliness.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Cats are independent pets that often enjoy their own space.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Goldfish are popular pets for beginners, requiring relatively simple care.",
        metadata={"source": "fish-pets-doc"},
    ),
    Document(
        page_content="Parrots are intelligent birds capable of mimicking human speech.",
        metadata={"source": "bird-pets-doc"},
    ),
    Document(
        page_content="Rabbits are social animals that need plenty of space to hop around.",
        metadata={"source": "mammal-pets-doc"},
    ),
]

vectorstore = Chroma.from_documents(documents=documents, embedding=OpenAIEmbeddings())

retriever = RunnableLambda(vectorstore.similarity_search).bind(
    k=1
)  # k=1 vector sonuclarinda ilk eslesen sonucu kullanmasi icin. digerini yoksayar

llm = ChatOpenAI(model="gpt-3.5-turbo")

message = """
Answer this question using the provided context only.
{question}

Context: {context}
"""

prompt = ChatPromptTemplate.from_messages([("human", message)])

chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm

if __name__ == "__main__":
    response = chain.invoke("tell me about cats")
    print(response.content)
