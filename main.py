from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import validators
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound
import os
from dotenv import load_dotenv
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from googleapiclient.discovery import build
import time
import random

load_dotenv(".env.local")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

app = FastAPI()




# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
def search_youtube(query: str):
    try:
        youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
        request = youtube.search().list(
            q=query, part="snippet", type="video", maxResults=1
        )
        response = request.execute()

        if "items" in response and response["items"]:
            video_id = response["items"][0]["id"]["videoId"]
            return f"https://www.youtube.com/watch?v={video_id}"

        return None

    except Exception as e:
        print("Error fetching from YouTube API:", str(e))
        return None
# Define request body model
class SummarizationRequest(BaseModel):
    url: str
    input: str  # Add input field

# Define the prompt template for map step
map_prompt_template = """
Summarize the following content:
{text}
"""
map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])

# Define the prompt template for combine step
combine_prompt_template = """
Combine these summaries into a final summary of  words and just be specific with the answer that input ask in short words , considering this additional context: {input}
Summaries:  {text}
"""
combine_prompt = PromptTemplate (template=combine_prompt_template, input_variables=["text", "input"])

def get_youtube_video_id(url):
    # Extract video ID from URL
    video_id_match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
    if video_id_match:
        return video_id_match.group(1)
    raise ValueError("Invalid YouTube URL")

# Define a simple Document class if not provided by the library
class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}

@app.post("/summarize")
async def summarize_content(request: SummarizationRequest):
    # Validate input
    if not request.url.strip() or not request.input.strip():
        raise HTTPException(status_code=400, detail="Please provide the necessary information.")
    if not validators.url(request.url):
        raise HTTPException(status_code=400, detail="Invalid URL. It should be a YouTube video or website URL.")

    try:
        # Initialize text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            length_function=len
        )

        # Load content from YouTube or Website
        if "youtube.com" in request.url or "youtu.be" in request.url:
            video_id = get_youtube_video_id(request.url)
            transcript = None
            for attempt in range(5):  # Retry up to 5 times
                try:
                    transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
                    try:
                        transcript = transcript_list.find_transcript(['en'])
                    except NoTranscriptFound:
                        transcript = transcript_list.find_transcript(['hi'])
                    break  # Exit loop if successful
                except Exception as e:
                    if "Too Many Requests" in str(e):
                        wait_time = (2 ** attempt) + random.uniform(0, 1)
                        print(f"Rate limit hit, retrying in {wait_time:.2f} seconds...")
                        time.sleep(wait_time)
                    else:
                        raise HTTPException(status_code=500, detail=f"Error loading YouTube transcript: {str(e)}")
            if not transcript:
                raise HTTPException(status_code=500, detail="Failed to retrieve transcript after multiple attempts.")
            
            transcript_text = transcript.fetch()
            all_text = " ".join([entry["text"] for entry in transcript_text])
            chunks = text_splitter.split_text(all_text)
            docs = [Document(chunk, metadata={"source": request.url}) for chunk in chunks]
        else:
            loader = UnstructuredURLLoader(
                urls=[request.url], ssl_verify=False,
                headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"}
            )
            all_text = " ".join([doc.page_content for doc in loader.load()])
            chunks = text_splitter.split_text(all_text)
            docs = [Document(chunk, metadata={"source": request.url}) for chunk in chunks]

        # Debugging: Print the loaded documents
        print("Loaded Documents:", docs)

        # Initialize the LLM and chain
        llm = ChatGroq(model="Llama3-8b-8192", groq_api_key=GROQ_API_KEY)
        chain = load_summarize_chain(
            llm, 
            chain_type="map_reduce",
            map_prompt=map_prompt,
            combine_prompt=combine_prompt,
            verbose=True
        )
        
        # Run the chain with input
        output_summary = chain.run(input_documents=docs, input=request.input)

        return {"summary": output_summary}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/test")
async def test_endpoint():
    return {"message": "Test successful"}

class SearchRequest(BaseModel):
    query: str

@app.post("/search")
async def search_youtube(request: SearchRequest):
    if not request.query:
        raise HTTPException(status_code=400, detail="Query is required")

    try:
        youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
        search_response = youtube.search().list(
            q=request.query,
            part="snippet",
            maxResults=1,
            type="video"
        ).execute()

        if not search_response["items"]:
            raise HTTPException(status_code=404, detail="No videos found")

        video_id = search_response["items"][0]["id"]["videoId"]
        video_link = f"https://www.youtube.com/watch?v={video_id}"

        return {"video_link": video_link}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch video: {str(e)}")
