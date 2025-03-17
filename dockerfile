# Use the official Python image as the base
FROM python:3.12-slim

# Set environment variables
ENV PORT=8080

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Download the NLTK dataset
RUN python -c "import nltk; nltk.download('punkt_tab')"

# Copy application code
COPY app /app
WORKDIR /app

# Expose the port Streamlit will run on
EXPOSE 8080

# Run the Streamlit app
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8080", "--server.address=0.0.0.0"]