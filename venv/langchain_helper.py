from langchain.vectorstores.faiss import FAISS
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()
embeddings = OpenAIEmbeddings(openai_api_key="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")


def create_vector_db_from_youtube_url(video_url: str) -> FAISS:
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)
    db = FAISS.from_documents(docs, embeddings)
    return db


def get_responses_from_query(db, query, k=5):
    docs = db.similarity_search(query, k=k)
    docs_page_content = "".join([d.page_content for d in docs])
    llm = OpenAI(openai_api_key="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", model="text-davinci-003")
    prompt = PromptTemplate(input_variables=["question", "docs"],
                            template="you are a helpful youtube assistant.answer this question {question} by searching the following video transcript {docs} and if you don't have enough information say 'I don't know'")
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(question=query, docs=docs_page_content)
    response = response.replace("\n", "")
    print("Response:", response)
    return response
