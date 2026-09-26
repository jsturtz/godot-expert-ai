FROM python:3.14-slim
RUN pip install uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen
COPY . .
EXPOSE 2024
CMD ["uv", "run", "langgraph", "dev", "--host", "0.0.0.0", "--no-browser"]