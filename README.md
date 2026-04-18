Overview
- RAG chatbot using groq
- allows user to upload the document
- ask questions in context of the provide document


RAG (retrival augmented generation)
- docs are splited in chunks 
- chunks are converted to embedddings
- store in vector db
- llm generate answer based on context only

GROQ api setup
- create a .env file
- generate your groq api key
- add to your .env file


task.ipynb - task(1,6),task (9,11)
app.py - task(7,8)


how to run 
- install depedencies
- run all cells of task.ipynb notebook
- run streamlit run app.py
