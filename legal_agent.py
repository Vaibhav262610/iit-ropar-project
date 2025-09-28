# Mergers and Acquisition agent
import os
import uuid
import pathway as pw
# Use fastapi for the API
from flask import request
from flask import Flask
app = Flask(__name__)

def agent():
    pass

# API for ingesting company documents
@app.route("/submit", methods=["POST"])
def ingest():
    company_a = {}
    company_b = {}
    company_a['legal'] = request.form.get("company_a_legal")
    company_b['legal'] = request.form.get("company_b_legal")
    company_a['financial'] = request.form.get("company_a_financial")
    company_b['financial'] = request.form.get("company_b_financial")
    user_instructions = request.form.get("user_instructions")
    # Store company documents in vector store by saving them in MA folder
    # Create new folder in MA folder with unique id
    MA_folder = f"MA/{uuid.uuid4()}"
    os.makedirs(MA_folder)
    # Save files to the folder
    with open(f"MA/{uuid.uuid4()}/company_a_legal.txt", "w") as f:
        f.write(company_a['legal'])
    with open(f"MA/{uuid.uuid4()}/company_b_legal.txt", "w") as f:
        f.write(company_b['legal'])
    with open(f"MA/{uuid.uuid4()}/company_a_financial.txt", "w") as f:
        f.write(company_a['financial'])
    with open(f"MA/{uuid.uuid4()}/company_b_financial.txt", "w") as f:
        f.write(company_b['financial'])
    data_sources = []
    data_sources.append(
        pw.io.fs.read(
            MA_folder,
            format="binary",
            mode="streaming",
            with_metadata=True,
        )
    )

def retrieve():
    # Yaha kaise retrieve karna hai documents dekh lo I am not sure
    pass

def parse_outline():
    # Import outline documents in MA folder
    # open term sheet
    with open("MA/term_sheet.txt", "r") as f:
        term_sheet = f.read()
        # Convert it into format string for LLM prompt
    # open defeinitive agreement
    with open("MA/definitive_agreement.txt", "r") as f:
        definitive_agreement = f.read()
        # Convert it into format string for LLM prompt
    # open merger agreement
    with open("MA/letter_of_intent.txt", "r") as f:
        letter_of_intent = f.read()
        # Convert it into format string for LLM prompt

def generate_documents():
    # Generate documents using LLM
    prompt_term_sheet = ""
    prompt_definitive_agreement = ""
    prompt_letter_of_intent = ""
    # Generate term sheet, definitive agreement and letter of intent

def generate_insights():
    # Jayesh ke dibbe me headings hain, usko according insights generate karna hai
    # Generate insights using LLM
    # This will be streamed through websockets at the end of processing

# This will be final documents delivered to user
def send_documents():
    # Send generated documents to user
    pass

Request (includes uploads) --> Thinking --> Generating Documents --> Sending Documents --> Streaming Insights