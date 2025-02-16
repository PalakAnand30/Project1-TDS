# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

COPY . /app

# Install any needed packages specified in requirements.txt

RUN pip install --no-cache-dir --upgrade pip && \
    pip install -r requirements.txt


# Expose the port FastAPI runs on
EXPOSE 8000


# Command to run the FastAPI application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
