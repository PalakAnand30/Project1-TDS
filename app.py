import os
import subprocess
import requests
import json
import calendar
import datetime
import shutil
import glob
import pytesseract
import pandas as pd
import sqlite3
import markdown
import duckdb
import re
import sqlite3
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from PIL import Image

# Load environment variables
load_dotenv()
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")
AIPROXY_URL = "http://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

if not AIPROXY_TOKEN:
    raise ValueError("AIPROXY_TOKEN is not set in the environment variables.")

# Initialize FastAPI app
app = FastAPI()

# 🔹 Task A1: Install UV and Run Datagen
SCRIPT_URL = "https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py"
SCRIPT_PATH = "datagen.py"

import shutil
import subprocess
from fastapi import HTTPException

import os
from fastapi import HTTPException

SAFE_ROOT = "/tmp/data/data/"  # This is where the actual files are located

def resolve_path(path: str):
    """Dynamically resolve paths, mapping /data/xyz → /tmp/data/data/xyz."""
    if path.startswith("/data/"):
        return path.replace("/data/", SAFE_ROOT)
    return path  # If already correct, return as is

def reverse_resolve_path(actual_path: str):
    """Convert actual stored path back to expected `/data/xyz` format."""
    if actual_path.startswith(SAFE_ROOT):
        return actual_path.replace(SAFE_ROOT, "/data/")
    return actual_path  # If already correct, return as is

# ✅ Prevent file deletions globally
def safe_remove(path: str):
    raise HTTPException(status_code=403, detail="File deletion is not allowed.")

os.remove = safe_remove  
os.rmdir = safe_remove 

# Required dependencies (Python packages + Prettier)
REQUIRED_PACKAGES = ["pillow", "faker", "uv"]

def install_dependencies():
    """Ensures that all required Python packages and global tools (like Prettier) are installed."""
    missing_packages = []
    
    pip_executable = shutil.which("pip3") or shutil.which("pip") or "python3 -m pip"

    # Use a virtual environment to avoid "externally-managed-environment" errors
    venv_path = "/tmp/myenv"
    
    if not os.path.exists(venv_path):
        subprocess.run(["python3", "-m", "venv", venv_path], check=True)
    
    pip_executable = os.path.join(venv_path, "bin", "pip")

    for package in REQUIRED_PACKAGES:
        try:
            subprocess.run([pip_executable, "install", package], check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError:
            missing_packages.append(package)

    if missing_packages:
        print(f"🔹 Installing missing Python packages in virtual environment: {', '.join(missing_packages)}...")
        try:
            subprocess.run([pip_executable, "install"] + missing_packages, check=True)
        except subprocess.CalledProcessError as e:
            print("❌ Failed to install required Python dependencies:", e.stderr)
            raise HTTPException(status_code=500, detail=f"Failed to install required dependencies: {e.stderr}")
    else:
        print("✅ All required Python packages are already installed.")

def download_script():
    """Fetches the datagen.py script and saves it locally."""
    response = requests.get(SCRIPT_URL)
    if response.status_code == 200:
        with open(SCRIPT_PATH, "w", encoding="utf-8") as file:
            file.write(response.text)
    else:
        raise HTTPException(status_code=500, detail="Failed to download datagen.py script")

def install_dependencies_and_execute_script(user_email: str):
    """Installs dependencies and runs datagen.py with the given email."""
    
    install_dependencies()  

    if not os.path.exists(SCRIPT_PATH):
        download_script()

    data_root = resolve_path("/data/")  # ✅ Secure the data path

    try:
        result = subprocess.run(
            ["python3", SCRIPT_PATH, user_email, "--root", data_root], 
            capture_output=True, text=True, check=True
        )
        return {"stdout": result.stdout, "stderr": result.stderr, "data_root": data_root}
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Execution failed: {e.stderr}")

# Task A2
def format_markdown():
    """
    Formats the contents of `/data/format.md` using Prettier (v3.4.2), updating the file in-place.
    Returns the formatted content of the file.
    """

    file_path = resolve_path("/data/format.md")  # Automatically resolve file path

    # ✅ Check if the file exists before formatting
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File `/data/format.md` not found.")

    try:
        # ✅ Run Prettier to format the file
        result = subprocess.run(
            ["npx", "prettier@3.4.2", "--write", file_path], 
            capture_output=True, text=True, check=True
        )

        # ✅ Verify if Prettier modified the file
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Prettier failed: {result.stderr}")

    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Prettier is not installed. Run: `npm install -g prettier`")
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Prettier formatting failed: {e.stderr}")

    # ✅ Read and return the formatted content
    with open(file_path, "r", encoding="utf-8") as file:
        formatted_content = file.read()

    return {
        "message": f"Markdown file `{reverse_resolve_path(file_path)}` formatted successfully.",
        "formatted_content": formatted_content
    }

# ✅ Task A3: Counting Weekday Occurrences
def count_weekday_occurrences(day_of_week: str, output_filename: str = None):
    """Counts a specific weekday in `dates.txt` and writes to a user-defined output file."""
    
    source_file = resolve_path("/data/dates.txt")  # ✅ Dynamic path resolution
    
    # ✅ If output_filename is not provided, default it
    if output_filename is None:
        output_filename = f"/data/dates-{day_of_week.lower()}.txt"
    
    output_file = resolve_path(output_filename)  # ✅ Dynamic output path

    if not os.path.exists(source_file):
        raise HTTPException(status_code=404, detail="dates.txt not found.")

    try:
        weekday_number = list(calendar.day_name).index(day_of_week.capitalize())
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid weekday name.")

    count = 0
    with open(source_file, "r", encoding="utf-8") as file:
        for line in file:
            try:
                date_obj = datetime.datetime.strptime(line.strip(), "%Y-%m-%d")
                if date_obj.weekday() == weekday_number:
                    count += 1
            except ValueError:
                continue  

    with open(output_file, "w", encoding="utf-8") as file:
        file.write(str(count))

    return {
        "message": f"Count of {day_of_week}s written to `{reverse_resolve_path(output_file)}`.",
        "count": count
    }


# task A4
def sort_contacts(primary_sort: str = "last_name", secondary_sort: str = "first_name"):
    """Sorts contacts in /data/contacts.json and writes the sorted result to /data/contacts-sorted.json.
    
    Args:
        primary_sort (str): The primary sorting key (default: "last_name").
        secondary_sort (str): The secondary sorting key (default: "first_name").
    """

    # 🔹 Convert `/data/contacts.json` → `/tmp/data/contacts.json`
    source_file = resolve_path("/data/contacts.json")
    output_file = resolve_path("/data/contacts-sorted.json")

    # ✅ Check if contacts.json exists
    if not os.path.exists(source_file):
        raise HTTPException(status_code=404, detail="contacts.json not found in /tmp/data/.")

    # 🔹 Read JSON file
    try:
        with open(source_file, "r", encoding="utf-8") as file:
            contacts = json.load(file)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format in contacts.json.")

    # ✅ Ensure contacts is a list of dictionaries
    if not isinstance(contacts, list) or not all(isinstance(contact, dict) for contact in contacts):
        raise HTTPException(status_code=400, detail="contacts.json must contain an array of objects.")

    # ✅ Check if required keys exist
    for contact in contacts:
        if primary_sort not in contact or secondary_sort not in contact:
            raise HTTPException(
                status_code=400,
                detail=f"Missing '{primary_sort}' or '{secondary_sort}' key in some contacts."
            )

    # 🔹 Sort contacts based on user preference
    sorted_contacts = sorted(contacts, key=lambda x: (x[primary_sort].lower(), x[secondary_sort].lower()))

    # ✅ Write sorted contacts to output file
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(sorted_contacts, file, indent=4)

    return {"message": f"Contacts sorted by {primary_sort}, then {secondary_sort} and saved to contacts-sorted.json."}

# task A5
def extract_recent_log_entries(position: str = "first", count: int = 10):
    """Extracts the first line from a user-specified number of recent .log files in /data/logs/
    
    Args:
        position (str): The part of the sorted logs to extract (default: "first").
                       - "first" → first N logs
                       - "last" → last N logs
                       - "middle" → middle N logs
        count (int): Number of log files to extract (default: 10).
    """

    # 🔹 Convert `/data/logs/` → `/tmp/data/logs/`
    logs_dir = resolve_path("/data/logs/")
    output_file = resolve_path("/data/logs-recent.txt")

    # ✅ Check if logs directory exists
    if not os.path.exists(logs_dir):
        raise HTTPException(status_code=404, detail="Logs directory not found in /tmp/data/. Please check Task 1 execution.")

    # 🔹 Get all .log files in the directory
    log_files = sorted(
        glob.glob(os.path.join(logs_dir, "*.log")), 
        key=os.path.getmtime, 
        reverse=True  # Sort by modification time (most recent first)
    )

    # ✅ Check if there are logs available
    if not log_files:
        raise HTTPException(status_code=404, detail="No log files found in /tmp/data/logs/.")

    # 🔹 Select files based on position
    total_logs = len(log_files)
    if position == "first":
        selected_logs = log_files[:count]  # Get first `count` logs
    elif position == "last":
        selected_logs = log_files[-count:]  # Get last `count` logs
    elif position == "middle":
        mid_start = max(0, (total_logs - count) // 2)
        selected_logs = log_files[mid_start:mid_start + count]
    else:
        raise HTTPException(status_code=400, detail="Invalid position value. Use 'first', 'last', or 'middle'.")

    # ✅ Extract first lines
    extracted_lines = []
    for log_file in selected_logs:
        try:
            with open(log_file, "r", encoding="utf-8") as file:
                first_line = file.readline().strip()  # Read first line
                extracted_lines.append(first_line)
        except Exception as e:
            continue  # Ignore files that can't be read

    # ✅ Write extracted lines to logs-recent.txt
    with open(output_file, "w", encoding="utf-8") as file:
        file.write("\n".join(extracted_lines))

    return {
        "message": f"Extracted {count} log entries ({position}) and saved to logs-recent.txt.",
        "files_processed": len(selected_logs),
        "output_file": output_file
    }

# ✅ Task A6: Extracting Markdown Titles
def extract_markdown_titles(output_filename: str = "/data/docs/index.json"):
    """Extracts the first H1 title from Markdown files in `/data/docs/` and stores results dynamically."""

    docs_dir = resolve_path("/data/docs/")
    index_file = resolve_path(output_filename)  

    if not os.path.exists(docs_dir):
        raise HTTPException(status_code=404, detail=f"Directory {docs_dir} not found.")

    index = {}
    
    # 🔹 Traverse Markdown files inside `/data/docs/`
    for root, _, files in os.walk(docs_dir):
        for file in files:
            if file.endswith(".md"):
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, docs_dir)  # ✅ Remove `/data/docs/` prefix

                with open(file_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                
                # 🔹 Extract **only the first H1** (`# Heading`)
                first_heading = None
                for line in lines:
                    if line.startswith("# "):  # ✅ Find first H1 heading
                        first_heading = line.strip("# ").strip()
                        break

                # ✅ Ensure we always store something (even if no H1 found)
                index[relative_path] = first_heading if first_heading else "No headings found"

    # ✅ Save JSON output correctly
    with open(index_file, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=4)

    return {
        "message": f"Markdown index created and saved to `{reverse_resolve_path(index_file)}`.",
        "index": index
    }

# task A7
def extract_email_addresses(extract_type="sender"):
    """Extracts sender, receiver, or both email addresses from email.txt and writes them to respective files."""
    
    email_file = resolve_path("/data/email.txt")  # ✅ Secure file access
    output_file = resolve_path(f"/data/email-{extract_type}.txt")  # ✅ Dynamic output file resolution

    if not os.path.exists(email_file):
        raise HTTPException(status_code=404, detail="email.txt not found.")

    with open(email_file, "r", encoding="utf-8") as f:
        email_content = f.read()

    sender_match = re.search(r'From:\s*".*?"\s*<(.+?)>', email_content)
    sender_email = sender_match.group(1) if sender_match else None

    receiver_match = re.search(r'To:\s*".*?"\s*<(.+?)>', email_content)
    receiver_email = receiver_match.group(1) if receiver_match else None

    extracted_data = {}

    if extract_type == "sender" and sender_email:
        extracted_data["sender_email"] = sender_email
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(sender_email)
    
    elif extract_type == "receiver" and receiver_email:
        extracted_data["receiver_email"] = receiver_email
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(receiver_email)

    elif extract_type == "both" and sender_email and receiver_email:
        extracted_data["sender_email"] = sender_email
        extracted_data["receiver_email"] = receiver_email
        with open(resolve_path("/data/email-sender.txt"), "w", encoding="utf-8") as f:
            f.write(sender_email)
        with open(resolve_path("/data/email-receiver.txt"), "w", encoding="utf-8") as f:
            f.write(receiver_email)

    else:
        raise HTTPException(status_code=500, detail="Could not extract the requested email address(es).")

    return {"message": "Email address(es) extracted successfully.", "data": extracted_data}

# Task A8
def extract_credit_card_number():
    """Extracts a credit card number from credit-card.png using OCR and writes it to credit-card.txt."""
    
    image_file = resolve_path("/data/credit_card.png")  # ✅ Secure file access
    output_file = resolve_path("/data/credit-card.txt")  # ✅ Secure output file

    if not os.path.exists(image_file):
        raise HTTPException(status_code=404, detail=f"{image_file} not found.")

    try:
        image = Image.open(image_file)
        ocr_result = pytesseract.image_to_string(image)  # Extract text from image
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process image: {str(e)}")

    match = re.search(r'\b(\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4})\b', ocr_result)
    
    if match:
        credit_card_number = match.group(1).replace(" ", "").replace("-", "")  # Remove spaces and hyphens
    else:
        raise HTTPException(status_code=500, detail="Could not extract a valid credit card number.")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(credit_card_number)

    return {"message": "Credit card number extracted successfully.", "credit_card_number": credit_card_number}

# Task A9
EMBEDDING_URL = "https://aiproxy.sanand.workers.dev/openai/v1/embeddings"
MODEL = "text-embedding-3-small"

# ✅ Task A9: Finding Most Similar Comments
def find_most_similar_comments(output_filename="/data/comments-similar.txt"):
    """
    Finds the most similar pair of comments in `/data/comments.txt` and writes them to a file.

    Args:
        output_filename (str): The output file where the most similar comments should be stored.

    Returns:
        dict: Confirmation message and extracted comments.
    """

    comments_file = resolve_path("/data/comments.txt")
    output_file = resolve_path(output_filename)

    if not os.path.exists(comments_file):
        raise HTTPException(status_code=404, detail=f"{comments_file} not found.")

    with open(comments_file, "r", encoding="utf-8") as f:
        comments = [line.strip() for line in f.readlines() if line.strip()]

    if len(comments) < 2:
        raise HTTPException(status_code=400, detail="Not enough comments to compare.")

    try:
        embeddings = get_text_embeddings(comments)
        most_similar_pair = compute_most_similar_pair(comments, embeddings)

        # 🔹 Write results to output file
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(most_similar_pair))

        return {
            "message": f"Most similar comments extracted and saved to `{reverse_resolve_path(output_file)}`.",
            "similar_comments": most_similar_pair
        }

    except HTTPException as e:
        return {"error": str(e)}

def compute_most_similar_pair(comments, embeddings):
    """
    Computes the most similar pair of comments based on cosine similarity.

    Args:
        comments (list): List of comment texts.
        embeddings (list): List of embedding vectors.

    Returns:
        tuple: The two most similar comments.
    """

    num_comments = len(comments)
    max_similarity = -1
    most_similar_pair = ("", "")

    # ✅ Compute pairwise cosine similarity
    for i in range(num_comments):
        for j in range(i + 1, num_comments):  # Avoid duplicate comparisons
            sim = cosine_similarity(embeddings[i], embeddings[j])
            if sim > max_similarity:
                max_similarity = sim
                most_similar_pair = (comments[i], comments[j])

    return most_similar_pair

def cosine_similarity(vec1, vec2):
    """
    Computes the cosine similarity between two vectors.

    Args:
        vec1 (list): First vector.
        vec2 (list): Second vector.

    Returns:
        float: Cosine similarity score.
    """
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)

    return dot_product / (norm_vec1 * norm_vec2)

# For embedding
def get_text_embeddings(text_list):
    """
    Calls OpenAI embedding API to generate vector embeddings for a list of texts.

    Args:
        text_list (list): List of strings to embed.

    Returns:
        list: List of embedding vectors (one per text).
    """

    headers = {
        "Authorization": f"Bearer {AIPROXY_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL,
        "input": text_list
    }

    try:
        response = requests.post(EMBEDDING_URL, headers=headers, json=payload)
        response_json = response.json()

        if "data" not in response_json:
            raise HTTPException(status_code=500, detail="Error retrieving embeddings from OpenAI API.")

        return [item["embedding"] for item in response_json["data"]]

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API Request Failed: {str(e)}")

# Task A10
# ✅ Task A10: Calculating Gold Ticket Sales
def calculate_gold_ticket_sales():
    db_file = resolve_path("/data/ticket-sales.db")
    output_file = resolve_path("/data/ticket-sales-gold.txt")

    if not os.path.exists(db_file):
        raise HTTPException(status_code=404, detail=f"{db_file} not found.")

    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute("SELECT SUM(units * price) FROM tickets WHERE type = 'Gold';")
        total_sales = cursor.fetchone()[0] or 0  # Ensure it's not None
        conn.close()

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(str(total_sales))

        return {"message": "Total sales for 'Gold' tickets calculated successfully.", "total_sales": total_sales}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database query failed: {str(e)}")

# Task B3
def fetch_and_save_api_data(api_url: str, output_filename: str):
    """
    Fetches data from a specified API and saves it to the given output file.

    Args:
        api_url (str): The API endpoint to fetch data from.
        output_filename (str): The filename (inside /data/) where the response should be saved.

    Returns:
        dict: Confirmation message and file location.
    """

    output_file = resolve_path(f"/data/{output_filename}")

    try:
        # ✅ Make a GET request to fetch API data
        response = requests.get(api_url)

        # ✅ Check if request was successful
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=f"Failed to fetch data from {api_url}")

        # ✅ Save the API response to the specified file
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(response.text)

        return {"message": "API data fetched and saved successfully.", "file": output_file}

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"API Request Failed: {str(e)}")

# Task B4
def clone_and_commit(repo_url: str, commit_message: str, filename: str = None, file_content: str = None):
    """
    Clones a Git repository, modifies or creates a file, and commits the changes.

    Args:
        repo_url (str): The URL of the Git repository to clone.
        commit_message (str): The commit message for the changes.
        filename (str, optional): The filename to create or modify inside the repo.
        file_content (str, optional): The content to write to the file.

    Returns:
        dict: Confirmation message and repository path.
    """

    # ✅ Define a temporary directory for cloning
    repo_name = os.path.basename(repo_url).replace(".git", "")
    repo_path = os.path.join("/tmp", repo_name)

    # ✅ Clean up if repo already exists
    if os.path.exists(repo_path):
        shutil.rmtree(repo_path)

    try:
        # ✅ Clone the repository
        subprocess.run(["git", "clone", repo_url, repo_path], check=True)

        # ✅ Change to the repository directory
        os.chdir(repo_path)

        # ✅ Modify or create a file if specified
        if filename and file_content:
            file_path = os.path.join(repo_path, filename)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(file_content)

            # ✅ Add the file to the commit
            subprocess.run(["git", "add", filename], check=True)

        # ✅ Commit the changes
        subprocess.run(["git", "commit", "-m", commit_message], check=True)

        return {"message": "Repository cloned and changes committed successfully.", "repo_path": repo_path}

    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Git command failed: {e.stderr}")

# Task B5
def run_sql_query(db_type: str, db_path: str, query: str):
    """
    Runs an SQL query on a given SQLite or DuckDB database.

    Args:
        db_type (str): The type of database ("sqlite" or "duckdb").
        db_path (str): The path to the database file.
        query (str): The SQL query to execute.

    Returns:
        dict: Query results in JSON format.
    """

    db_file = resolve_path(db_path)  # Ensure correct database path

    # ✅ Check if database file exists
    if not os.path.exists(db_file):
        raise HTTPException(status_code=404, detail=f"Database file {db_file} not found.")

    try:
        # ✅ Choose database type
        if db_type == "sqlite":
            conn = sqlite3.connect(db_file)
        elif db_type == "duckdb":
            conn = duckdb.connect(db_file)
        else:
            raise HTTPException(status_code=400, detail="Unsupported database type. Use 'sqlite' or 'duckdb'.")

        # ✅ Run query and fetch results
        df = pd.read_sql_query(query, conn)
        conn.close()

        return {"message": "Query executed successfully.", "results": df.to_dict(orient="records")}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SQL query failed: {str(e)}")

# Task B6
def scrape_website(url: str, output_filename: str, element_type: str = "p"):
    """
    Scrapes a website and extracts specific elements (default: paragraphs), saving the extracted text to a file.

    Args:
        url (str): The website URL to scrape.
        output_filename (str): The filename inside /data/ where the extracted content will be saved.
        element_type (str, optional): The type of HTML element to extract (e.g., 'p', 'h1', 'a', 'table').

    Returns:
        dict: Confirmation message and output file location.
    """

    output_file = resolve_path(f"/data/{output_filename}")

    try:
        # ✅ Fetch website content
        response = requests.get(url)
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=f"Failed to fetch website: {url}")

        # ✅ Parse HTML using BeautifulSoup
        soup = BeautifulSoup(response.text, "html.parser")

        # ✅ Extract specific elements (default: paragraphs)
        extracted_content = "\n".join([elem.get_text(strip=True) for elem in soup.find_all(element_type)])

        # ✅ Save extracted content
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(extracted_content)

        return {"message": "Website scraped successfully.", "file": output_file}

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch website: {str(e)}")



# Task B7
def compress_or_resize_image(input_filename: str, output_filename: str, max_size_kb: int = None, resize_to: tuple = None):
    """
    Compresses or resizes an image based on user input.

    Args:
        input_filename (str): The input image file inside /data/.
        output_filename (str): The output image file inside /data/.
        max_size_kb (int, optional): Maximum file size in KB after compression.
        resize_to (tuple, optional): New dimensions as (width, height).

    Returns:
        dict: Confirmation message with output file details.
    """

    input_file = resolve_path(f"/data/{input_filename}")
    output_file = resolve_path(f"/data/{output_filename}")

    # ✅ Check if image file exists
    if not os.path.exists(input_file):
        raise HTTPException(status_code=404, detail=f"Image file {input_file} not found. Run Task A1 first.")

    try:
        with Image.open(input_file) as img:
            # ✅ Resize if dimensions are provided
            if resize_to:
                img = img.resize(resize_to, Image.ANTIALIAS)

            # ✅ Compress if max_size_kb is provided
            quality = 95  # Start with high quality
            if max_size_kb:
                while True:
                    img.save(output_file, quality=quality, optimize=True)
                    if os.path.getsize(output_file) / 1024 <= max_size_kb or quality <= 10:
                        break
                    quality -= 5  # Reduce quality in steps of 5%

            else:
                img.save(output_file, quality=90, optimize=True)

        return {"message": "Image processed successfully.", "output_file": output_file}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image processing failed: {str(e)}")

# Task B8
def transcribe_audio(input_filename: str, output_filename: str):
    """
    Transcribes an MP3 audio file using GPT-4o Mini and saves the transcription.

    Args:
        input_filename (str): The input MP3 file inside /data/.
        output_filename (str): The output text file inside /data/.

    Returns:
        dict: Confirmation message with output file details.
    """

    input_file = resolve_path(f"/data/{input_filename}")
    output_file = resolve_path(f"/data/{output_filename}")

    # ✅ Check if audio file exists
    if not os.path.exists(input_file):
        raise HTTPException(status_code=404, detail=f"Audio file {input_file} not found. Upload an MP3 file first.")

    # ✅ Send the audio file to OpenAI API
    headers = {"Authorization": f"Bearer {AIPROXY_TOKEN}"}
    files = {"file": open(input_file, "rb")}
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "Transcribe the provided audio accurately."},
            {"role": "user", "content": "Here's the audio file for transcription."}
        ]
    }

    try:
        response = requests.post(AIPROXY_URL, headers=headers, files=files, json=payload)
        response_json = response.json()

        # ✅ Extract transcription from API response
        if "choices" in response_json and response_json["choices"]:
            transcription = response_json["choices"][0]["message"]["content"]
        else:
            raise HTTPException(status_code=500, detail="Transcription failed: Invalid API response.")

        # ✅ Save transcription
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(transcription)

        return {"message": "Audio transcribed successfully.", "output_file": output_file}

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"API request failed: {str(e)}")

# Task B9
def convert_markdown_to_html(input_filename: str, output_filename: str):
    """
    Converts a Markdown (.md) file to HTML and saves the output.

    Args:
        input_filename (str): The name of the Markdown file inside /data/.
        output_filename (str): The name of the output HTML file inside /data/.

    Returns:
        dict: Confirmation message with output file details.
    """

    input_file = resolve_path(f"/data/{input_filename}")
    output_file = resolve_path(f"/data/{output_filename}")

    # ✅ Check if Markdown file exists
    if not os.path.exists(input_file):
        raise HTTPException(status_code=404, detail=f"Markdown file {input_file} not found.")

    try:
        # ✅ Read Markdown content
        with open(input_file, "r", encoding="utf-8") as f:
            markdown_content = f.read()

        # ✅ Convert Markdown to HTML
        html_content = markdown.markdown(markdown_content)

        # ✅ Save HTML output
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_content)

        return {"message": "Markdown converted to HTML successfully.", "output_file": output_file}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Markdown conversion failed: {str(e)}")

# Task B10
def filter_csv(input_filename: str, column: str, value: str):
    """
    Filters a CSV file based on a given column and value, and returns JSON data.

    Args:
        input_filename (str): The name of the CSV file inside /data/.
        column (str): The column to filter by.
        value (str): The value to filter for.

    Returns:
        dict: Filtered data in JSON format.
    """

    input_file = resolve_path(f"/data/{input_filename}")

    # ✅ Check if CSV file exists
    if not os.path.exists(input_file):
        raise HTTPException(status_code=404, detail=f"CSV file {input_file} not found.")

    try:
        # ✅ Load CSV into DataFrame
        df = pd.read_csv(input_file)

        # ✅ Validate column existence
        if column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{column}' not found in {input_filename}. Available columns: {list(df.columns)}")

        # ✅ Filter DataFrame
        filtered_df = df[df[column] == value]

        # ✅ Convert to JSON
        filtered_json = filtered_df.to_dict(orient="records")

        return {"message": "CSV filtered successfully.", "filtered_data": filtered_json}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CSV processing failed: {str(e)}")


# 🔹 OpenAI Function Calling
def call_openai_function_calling(user_request: str, available_functions: list):
    """
    Uses OpenAI's function calling to determine which task to run.

    Args:
        user_request (str): The user's natural language task description.
        available_functions (list): A list of function descriptions for OpenAI.

    Returns:
        dict: A dictionary with 'task' (function name) and 'parameters' (dict).
    """
    headers = {
        "Authorization": f"Bearer {AIPROXY_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": user_request}],
        "functions": available_functions,
        "function_call": "auto"
    }

    try:
        response = requests.post(AIPROXY_URL, headers=headers, json=payload)
        response_json = response.json()

        # 🔹 Ensure OpenAI API responded with valid data
        if "choices" not in response_json or not response_json["choices"]:
            print("❌ OpenAI API Response Error:", response_json)  # Log error response
            raise HTTPException(status_code=500, detail=f"OpenAI API Error: {response_json}")

        response_message = response_json["choices"][0]["message"]

        if "function_call" in response_message:
            function_name = response_message["function_call"]["name"]
            arguments = json.loads(response_message["function_call"]["arguments"])
            return {"task": function_name, "parameters": arguments}

        raise HTTPException(status_code=400, detail="No valid function identified.")

    except requests.exceptions.RequestException as e:
        print("❌ OpenAI API Request Failed:", e)
        raise HTTPException(status_code=500, detail=f"OpenAI API Request Failed: {str(e)}")

    except json.JSONDecodeError as e:
        print("❌ OpenAI API Response is Not JSON:", e)
        raise HTTPException(status_code=500, detail="OpenAI API Error: Response not in JSON format.")

    except Exception as e:
        print("❌ Unexpected OpenAI API Error:", e)
        raise HTTPException(status_code=500, detail=f"Unexpected Error: {str(e)}")

# 🔹 Task Functions
TASK_FUNCTIONS = {
    "install_uv_and_run_datagen": {
        "description": "Installs dependencies (including uv, pillow, faker) and runs datagen.py with the provided user email.",
        "parameters": {
            "type": "object",
            "properties": {"user_email": {"type": "string", "description": "User email for the script."}},
            "required": ["user_email"]
        },
        "function": lambda params: install_dependencies_and_execute_script(params["user_email"])  # ✅ Updated
    },
    "format_markdown": {
        "description": "Formats the contents of /tmp/data/format.md using prettier@3.4.2, updating the file in-place.",
        "parameters": {"type": "object", "properties": {}, "required": []},
        "function": lambda _: format_markdown()
    },
    "count_weekday_occurrences": {
    "description": "Counts occurrences of a specified weekday in /data/dates.txt and writes the count to /data/dates-{day}.txt.",
    "parameters": {
        "type": "object",
        "properties": {
            "day_of_week": {
                "type": "string",
                "description": "The day of the week to count occurrences of (e.g., 'Wednesday')."
            }
        },
        "required": ["day_of_week"]
    },
    "function": lambda params: count_weekday_occurrences(params["day_of_week"])
}, "sort_contacts": {
    "description": "Sorts the contacts in /data/contacts.json based on user-specified order and writes them to /data/contacts-sorted.json.",
    "parameters": {
        "type": "object",
        "properties": {
            "primary_sort": {
                "type": "string",
                "description": "The primary sorting key (default: 'last_name')."
            },
            "secondary_sort": {
                "type": "string",
                "description": "The secondary sorting key (default: 'first_name')."
            }
        },
        "required": ["primary_sort", "secondary_sort"]
    },
    "function": lambda params: sort_contacts(params["primary_sort"], params["secondary_sort"])
}, "extract_recent_log_entries": {
    "description": "Extracts the first line from a user-specified number of recent .log files in /data/logs/ and writes them to /data/logs-recent.txt.",
    "parameters": {
        "type": "object",
        "properties": {
            "position": {
                "type": "string",
                "description": "Which part of the logs to extract (options: 'first', 'last', 'middle'). Default: 'first'."
            },
            "count": {
                "type": "integer",
                "description": "How many log files to process. Default: 10."
            }
        },
        "required": ["position", "count"]
    },
    "function": lambda params: extract_recent_log_entries(params["position"], params["count"])
}, "extract_markdown_titles": {
    "description": "Extracts headings from Markdown files in /data/docs/ and creates an index.",
    "parameters": {
        "type": "object",
        "properties": {
            "heading_levels": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "Which heading levels to extract (e.g., [1] for H1, [2] for H2, [1,2] for both H1 and H2).",
                "default": [1]
            },
            "num_headings": {
                "type": "integer",
                "description": "How many headings to extract per file (None for all available headings).",
                "default": None
            },
            "output_filename": {
                "type": "string",
                "description": "The output filename where the index will be saved. Defaults to /data/docs/index.json.",
                "default": "/data/docs/index.json"
            }
        },
        "required": []
    },
    "function": lambda params: extract_markdown_titles(
    params.get("output_filename", "/data/docs/index.json")
)
},"extract_email_addresses":{
    "description": "Extracts sender, receiver, or both email addresses from /data/email.txt and writes them to respective files.",
    "parameters": {
        "type": "object",
        "properties": {
            "extract_type": {
                "type": "string",
                "enum": ["sender", "receiver", "both"],
                "description": "Which email to extract: 'sender' (default), 'receiver', or 'both'."
            }
        },
        "required": ["extract_type"]
    },
    "function": lambda params: extract_email_addresses(params["extract_type"])
}, "extract_credit_card_number":{
    "description": "Extracts a credit card number from /data/credit-card.png and writes it to /data/credit-card.txt.",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": []
    },
    "function": lambda _: extract_credit_card_number()
}, "find_most_similar_comments": {
    "description": "Finds the most similar pair of comments in /data/comments.txt using embeddings and writes them to /data/comments-similar.txt.",
    "parameters": {
        "type": "object",
        "properties": {
            "output_filename": {
                "type": "string",
                "description": "The filename where the most similar comments will be saved. Default: /data/comments-similar.txt.",
                "default": "/data/comments-similar.txt"
            }
        },
        "required": []
    },
    "function": lambda params: find_most_similar_comments(
        params.get("output_filename", "/data/comments-similar.txt")
    )
}, "calculate_gold_ticket_sales":{
    "description": "Calculates total sales for 'Gold' tickets in /data/ticket-sales.db and writes the result to /data/ticket-sales-gold.txt.",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": []
    },
    "function": lambda _: calculate_gold_ticket_sales()
}, "fetch_and_save_api_data":{
    "description": "Fetches data from an API and saves it to a specified file.",
    "parameters": {
        "type": "object",
        "properties": {
            "api_url": {"type": "string", "description": "The API URL to fetch data from."},
            "output_filename": {"type": "string", "description": "The filename to save the API response (inside /data/)."}
        },
        "required": ["api_url", "output_filename"]
    },
    "function": lambda params: fetch_and_save_api_data(params["api_url"], params["output_filename"])
}, "clone_and_commit":{
    "description": "Clones a Git repository, makes a change, and commits it.",
    "parameters": {
        "type": "object",
        "properties": {
            "repo_url": {"type": "string", "description": "The URL of the Git repository to clone."},
            "commit_message": {"type": "string", "description": "The commit message for the changes."},
            "filename": {"type": "string", "description": "The filename to modify or create (optional)."},
            "file_content": {"type": "string", "description": "The content to write to the file (optional)."}
        },
        "required": ["repo_url", "commit_message"]
    },
    "function": lambda params: clone_and_commit(params["repo_url"], params["commit_message"], params.get("filename"), params.get("file_content"))
}, "run_sql_query":{
    "description": "Runs an SQL query on a SQLite or DuckDB database.",
    "parameters": {
        "type": "object",
        "properties": {
            "db_type": {"type": "string", "description": "The type of database ('sqlite' or 'duckdb')."},
            "db_path": {"type": "string", "description": "The path to the database file."},
            "query": {"type": "string", "description": "The SQL query to execute."}
        },
        "required": ["db_type", "db_path", "query"]
    },
    "function": lambda params: run_sql_query(params["db_type"], params["db_path"], params["query"])
}, "compress_or_resize_image":{
    "description": "Compresses or resizes an image inside /data/.",
    "parameters": {
        "type": "object",
        "properties": {
            "input_filename": {"type": "string", "description": "The name of the input image file (inside /data/)."},
            "output_filename": {"type": "string", "description": "The name of the output file (inside /data/)."},
            "max_size_kb": {"type": "integer", "description": "Maximum file size in KB after compression (optional)."},
            "resize_to": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "Resize dimensions as [width, height] (optional)."
            }
        },
        "required": ["input_filename", "output_filename"]
    },
    "function": lambda params: compress_or_resize_image(
        params["input_filename"],
        params["output_filename"],
        params.get("max_size_kb"),
        tuple(params["resize_to"]) if "resize_to" in params else None
    )
}, "transcribe_audio":{
    "description": "Transcribes an MP3 file using GPT-4o Mini and saves the text.",
    "parameters": {
        "type": "object",
        "properties": {
            "input_filename": {"type": "string", "description": "The name of the input MP3 file (inside /data/)."},
            "output_filename": {"type": "string", "description": "The name of the output text file (inside /data/)."}
        },
        "required": ["input_filename", "output_filename"]
    },
    "function": lambda params: transcribe_audio(params["input_filename"], params["output_filename"])
}, "convert_markdown_to_html":{
    "description": "Converts a Markdown (.md) file to HTML and saves it.",
    "parameters": {
        "type": "object",
        "properties": {
            "input_filename": {"type": "string", "description": "The name of the input Markdown file (inside /data/)."},
            "output_filename": {"type": "string", "description": "The name of the output HTML file (inside /data/)."}
        },
        "required": ["input_filename", "output_filename"]
    },
    "function": lambda params: convert_markdown_to_html(params["input_filename"], params["output_filename"])
}, "filter_csv":{
    "description": "Filters a CSV file based on a column and value, returning JSON data.",
    "parameters": {
        "type": "object",
        "properties": {
            "input_filename": {"type": "string", "description": "The name of the input CSV file (inside /data/)."},
            "column": {"type": "string", "description": "The column to filter by."},
            "value": {"type": "string", "description": "The value to filter for."}
        },
        "required": ["input_filename", "column", "value"]
    },
    "function": lambda params: filter_csv(params["input_filename"], params["column"], params["value"])
}, "scrape_website":{
    "description": "Scrapes a website and extracts specific elements, saving them to a file.",
    "parameters": {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "The website URL to scrape."},
            "output_filename": {"type": "string", "description": "The filename where extracted data is saved (inside /data/)."},
            "element_type": {
                "type": "string",
                "description": "The type of HTML element to extract (default: 'p' for paragraphs).",
                "default": "p"
            }
        },
        "required": ["url", "output_filename"]
    },
    "function": lambda params: scrape_website(
        params["url"], params["output_filename"], params.get("element_type", "p")
    )
}
}

available_functions = [{"name": name, "description": task["description"], "parameters": task["parameters"]} for name, task in TASK_FUNCTIONS.items()]

@app.post("/run")
async def process_task(task: str):
    """API endpoint to process user requests dynamically using OpenAI function calling."""
    response = call_openai_function_calling(task, available_functions)  # No change to NLP handling
    task_name = response["task"]
    params = response["parameters"]

    if task_name in TASK_FUNCTIONS:
        return {"message": "Task executed successfully", "output": TASK_FUNCTIONS[task_name]["function"](params)}

    raise HTTPException(status_code=400, detail=f"Unknown task. Available tasks: {list(TASK_FUNCTIONS.keys())}")

BASE_DIR = "/tmp/data/data"  # Update this if needed

@app.get("/read")
def get_file(path: str):
    """Dynamically read files from the correct directory"""
    file_path = resolve_path(path)  # ✅ Ensures correct mapping

    print(f"Checking file path: {file_path}")  # Debugging log

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

    return FileResponse(file_path)