# MOSDAC Chatbot UI

Simple React + TypeScript front-end for interacting with the MOSDAC RAG chatbot backend.

## Prerequisites

* Node.js ≥ 18

## Quick Start

```bash
cd frontend
npm install  # or pnpm install / yarn
npm run dev
```

The dev server runs at http://localhost:5173 and proxies API requests (`/chat`) to http://localhost:8000 (configured in `vite.config.ts`). Ensure the backend FastAPI service is running on that port.

## Build for Production

```bash
npm run build
```

Output will be in `dist/`, which can be served by any static server or integrated into the backend container.