from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from typing import List
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

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

if __name__ == "__main__":
    embedding = OpenAIEmbeddings().embed_query(
        "dog"
    )  # "dog" querysine sahip bir vector olusturduk
    print(
        vectorstore.similarity_search_by_vector(embedding=embedding)
    )  # olusturulan vectoru, vectorstore uzerinden similarity aramasinda kullandik.
    print(vectorstore.similarity_search("cat"))  # kelime bazli arama
    print(
        vectorstore.similarity_search_with_score("rabbit")
    )  # burada kelime bazli arama yapilir, ekstra olarak eslesme skorlari da sonuclara eklenir
