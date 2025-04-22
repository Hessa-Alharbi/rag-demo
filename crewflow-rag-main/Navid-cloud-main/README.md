# Arabic RAG (Retrieval-Augmented Generation)

A modern document question-answering system optimized for Arabic content, built with React, FastAPI, and vLLM.



## Features

- 🌐 Modern React frontend with RTL support
- 📄 Document upload with drag-and-drop
- 💬 Interactive chat interface
- 🎨 Dark/light theme support
- 📱 Mobile-responsive design
- 🔍 Semantic search using ChromaDB
- 🤖 Advanced text generation using CohereForAI/c4ai-command-r7b-12-2024
- 🐳 Containerized with Docker for easy deployment

## Architecture

The project consists of three main components:

1. **Frontend**: React + TypeScript + Vite
   - Material UI components with RTL support
   - Real-time chat interface
   - Document management
   - Responsive design

2. **Backend**: Python + FastAPI
   - Document processing and storage
   - Vector embeddings with ChromaDB
   - RESTful API endpoints

3. **Model Server**: vLLM
   - High-performance inference using CohereForAI/c4ai-command-r7b-12-2024
   - OpenAI-compatible API interface

## Prerequisites

- Docker and Docker Compose
- NVIDIA GPU with CUDA support
- NVIDIA Container Toolkit

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/Navid-Gen-AI/arabic-rag.git
   cd arabic-rag
   ```

2. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env file with your configuration
   ```

3. Build and start the services:
   ```bash
   docker compose up --build
   ```

4. Access the application:
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - vLLM Server: http://localhost:8080

## API Endpoints

### Backend API

- `POST /api/documents/upload`
  - Upload one or more documents for processing
  - Supports PDF, DOCX, and TXT files
  - Max file size: 10MB

- `POST /api/query`
  - Query the system with a question
  - Returns answer and source documents

## Development

### Frontend Development

```bash
cd frontend
npm install
npm run dev
```

### Backend Development

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Project Structure

```
├── docker-compose.yml
│
├── frontend/
│   ├── Dockerfile
│   ├── src/
│   │   ├── App.tsx
│   │   ├── services/
│   │   └── ...
│   └── package.json
│
├── backend/
│   ├── Dockerfile
│   ├── app/
│   │   ├── main.py
│   │   ├── api/
│   │   ├── models/
│   │   └── services/
│   └── requirements.txt
│
└── .env.example
```

## Environment Variables

The project uses a single `.env` file in the root directory for all configuration:

- `API_URL`: Backend API URL (default: http://localhost:8000)
- `VLLM_API_URL`: vLLM service URL (default: http://vllm:8000)
- `HUGGING_FACE_HUB_TOKEN`: Your HuggingFace token (if needed for private models)

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [vLLM](https://github.com/vllm-project/vllm) for high-performance inference
- [CohereForAI](https://huggingface.co/CohereForAI/c4ai-command-r7b-12-2024) for the Arabic language model
- [ChromaDB](https://github.com/chroma-core/chroma) for vector storage
