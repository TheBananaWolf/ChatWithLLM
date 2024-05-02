# ChatWithLLM
A local web application allow you to chat with you file and the LLM

## How to use it
- Create .env and place the HuggingFace Api Key in that file like ```HUGGINGFACEHUB_API_TOKEN = ```
- Then run ```conda create -n ENV_NAME python=3.9```
- Then run ```pip install -r requirements.txt```
- Then run ```streamlit run app.py```
- Default is chat with the Llama 3.
- Chat with pdf first upload file at sidebar, then enter question at input area.
- Switch mode from chat with pdf to take with llm enter "not pdf" at input area.
- Switch mode from chat with take with llm to pdf enter "pdf" at input area.


## Pre-requirement
- Conda
- Pip
- Ollama
- Docker
