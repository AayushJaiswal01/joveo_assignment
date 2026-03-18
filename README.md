#  GitLab GenAI Chatbot

A Retrieval-Augmented Generation (RAG) chatbot designed to help employees and aspiring team members easily navigate and understand GitLab’s Handbook and Direction pages. 

## Live Application
You can access the deployed version of this chatbot here:
** https://joveoassignment-2v6gp37zif8bdiwh7h366d.streamlit.app**

##  Features
* **Conversational Memory:** Remembers chat history, allowing users to ask natural, seamless follow-up questions.
* **Source Transparency:** Extracts metadata from the vector database to explicitly list the exact URLs used to generate every answer.
* **Enterprise Guardrails:** A strict system prompt prevents the AI from answering general knowledge questions or writing code, keeping it focused strictly on GitLab documentation.
* **Hybrid Architecture:** Uses local HuggingFace embeddings (`all-MiniLM-L6-v2`) for fast, free, and limit-less data ingestion, paired with Google Gemini for high-quality text generation.

## Tech Stack
* **Frontend:** Streamlit
* **Framework:** LangChain (LCEL)
* **LLM:** Google Gemini (2.5 Flash)
* **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`)
* **Vector Database:** FAISS

##  How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AayushJaiswal01/joveo_assignment
   cd joveo_assignment
2. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt

3. **Set up your API Key:**
Create a folder named .streamlit in the root directory, and inside it create a file named secrets.toml. Add your Google Gemini API key:

GOOGLE_API_KEY = "your_api_key_here"

4. **Run the application:**
   ```bash
   streamlit run app.py
