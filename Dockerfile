FROM python:3.10-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y build-essential

# Copy project
COPY src ./src
COPY requirements.txt ./requirements.txt
COPY docker/entrypoint.sh ./entrypoint.sh
COPY README.md ./README.md

RUN chmod +x ./entrypoint.sh

# Install dependencies
RUN pip install uv && \
    uv pip install --system -r requirements.txt && \
    uv pip install --system -e src

ENV PYTHONPATH="/app/src"

# Expose Chainlit
EXPOSE 8000

ENTRYPOINT ["./entrypoint.sh"]
