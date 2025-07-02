# MOSDAC Chatbot UI

Simple React + TypeScript front-end for interacting with the MOSDAC RAG chatbot backend.

## Prerequisites

* Node.js ≥ 18
* npm (comes with Node) or pnpm / yarn

> The project uses React 18 with the new JSX transform and Vite. Development dependencies include `@types/react`, `@types/react-dom`, and `@types/uuid` to ensure TypeScript type safety.

## Install dependencies

```bash
cd frontend
npm install     # installs runtime + dev dependencies
```

If you are using `pnpm` or `yarn`, replace the command accordingly.  The command installs:

* `react` & `react-dom` v18.2 – runtime libraries
* `@types/react`, `@types/react-dom`, `@types/uuid` – TypeScript typings
* `vite` & `@vitejs/plugin-react` – build/dev tooling

> Note: If you previously installed React 18.3 beta, remove `node_modules` and reinstall to avoid mismatched type versions.

## Development mode

```bash
npm run dev
```

This starts Vite's dev-server at <http://localhost:5173>.  API calls to `/chat` are proxied to the backend running at <http://localhost:8000> (see `vite.config.ts`).  Ensure you have the FastAPI service running first:

```bash
uvicorn service.app:app --reload --port 8000
```

## Production build

```bash
npm run build
```

Static files will be emitted to `dist/`.  You can serve them with any static web server or copy them into a backend container.

---

### Troubleshooting

* **Type errors like "cannot find module 'react/jsx-runtime'"** – make sure you ran `npm install` which installs React's type declarations.
* **Port conflict on 5173** – change the port in `vite.config.ts`.