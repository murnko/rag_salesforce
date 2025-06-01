# RAG Salesforce Earnings Assistant

This is a Retrieval-Augmented Generation (RAG) application built to parse, embed, and query Salesforce earnings call transcripts using LangChain, FAISS, and Chainlit.

## Project Structure

```
.
└── src/
    ├── config/         # Configuration files and environment variables
    ├── data_ingestion/ # Data loading and vectorization modules
    ├── retrieval/      # Retriever implementation and graph processing
    ├── scripts/        # Vectorization and utility pipelines
    └── application/    # Chainlit web application
```

## Prerequisites

- Docker
- Python 3.10+
- Poetry (Python package manager)
- Required API keys (see Configuration section)

## Configuration

1. Create a `secrets.env` file in `src/config/` with the following variables:
   ```env
   OPENAI_API_KEY=
   ```

## Running with Docker

### Using Docker (Recommended)


1. Make the entrypoint script executable:
   ```bash
   chmod +x docker/entrypoint.sh
   ```

2. Build the Docker image:
   ```bash
   docker build -t rag-app .
   ```

3. Run the container:
   ```bash
   docker run --env-file src/config/secrets.env -p 8000:8000 rag-app
   ```

4. Access the application at `http://localhost:8000`

### Troubleshooting Docker

If you encounter a permission error like this:
```bash
ERROR: permission denied while trying to connect to the Docker daemon socket
```

Fix it by either:
1. Adding your user to the docker group:
   ```bash
   sudo usermod -aG docker ${USER}
   newgrp docker
   ```
2. Or running Docker commands with sudo:
   ```bash
   sudo docker build -t rag-app .
   sudo docker run --env-file src/config/secrets.env -p 8000:8000 rag-app
   ```

### Local Development

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/rag-salesforce-earnings.git
   cd rag-salesforce-earnings
   ```

2. Install dependencies:
   ```bash
   poetry install
   ```

3. Run the application:
   ```bash
   poetry run chainlit run src/application/app.py
   ```

## Usage

1. Download Salesforce earnings call transcripts
2. Wait for the processing and vectorization to complete
3. Start querying the data using natural language

## Development Plan

See [DEVELOPMENT_PLAN.txt](DEVELOPMENT_PLAN.txt)
