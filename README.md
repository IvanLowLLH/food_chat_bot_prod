# Food Chatbot
This repo contains code to build a chatbot for recommending food places in Singapore.

## Creating Databases
`gmap_scrap` folder contains code to generate vector (Chroma) and lexical (BM25) databases.
Once generated, copy the files to `app` folder.

## Set-up chatbot code
`app/streamlit_app.py` contains the main code of the chatbot app. By default, it uses
Llama 3.1 8B as the main LLM and for query re-write + re-formatting. Edit `bm25_file` and 
`chroma_path` variable for BM25 file and Chroma folder respectively.

If you have a Firebase database that you want to save queries into, can add it to `app` folder.
Otherwise, it is not needed.

## Docker
To run the chatbot, build the Docker image using:

```commandline
docker build -t <image-name> .
```
and then run it as a container using:
```commandline
docker run -p 8080:8080 <image-name>
```
