#%%
from re import search

from langchain_classic.chains.question_answering.map_reduce_prompt import messages
from pydantic_core.core_schema import none_schema
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
# from youtube_transcript_api import YouTubeTranscriptApi
from langchain_ollama import OllamaEmbeddings
import ollama

# from llm_project import vector_db
#%%
yt_apikey="Enter Your YOUTUBE API KEY"
#%%
!pip install youtube-transcript-api
#%%
from youtube_transcript_api import YouTubeTranscriptApi
#%%
def get_transcript(video_id):
    try :
        transcript_list=YouTubeTranscriptApi().fetch(video_id)
        transcript = " ".join(chunk.text for chunk in transcript_list)
        return transcript
    except TranscriptsDisabled:
        print("No Caption Available")
#%%
transcript=get_transcript("Q9FvF9FXcOs")
print(transcript)
#%%
splitter=RecursiveCharacterTextSplitter(chunk_size=300,chunk_overlap=50)
chunks=splitter.create_documents([transcript])
print(chunks)
#%%
embeddings=OllamaEmbeddings(model="qwen3-embedding:0.6b")
vector_db=FAISS.from_documents(chunks,embeddings)
#%%
retriver=vector_db.as_retriever(search_type="similarity",search_kwargs={"k":4})
#%%
prompt=PromptTemplate(template=""" You are a helpful assistant.
answer ONLY form the transcript provided
If the context is insufficient, just say i don't know.

{context}
question : {question}
"""
,input_variables=["context","question"]
)
#%%
question="why do humans wonder?"
founded_docs=retriver.invoke(question)
#%%
context_text="\n\n".join(text.page_content for text in founded_docs)
print(context_text)
#%%
final_prompt=prompt.invoke({"context":context_text,"question":question})
#%%
print(final_prompt)
#%%
from langchain_community.llms import Ollama
llm=Ollama(model="qwen3:4b")
#%%

#%%
chain = prompt | llm
#%%
response=chain.invoke({"context":context_text,"question":question})
#%%
print(response)

#%%
context_text
#%%
