# Vite frontend — build, then serve the static bundle with vite preview.
# Build context = repo root (so .dockerignore at root applies).
FROM node:20-slim

WORKDIR /app

COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install

COPY frontend ./
RUN npm run build

EXPOSE 5173
CMD ["npm", "run", "preview", "--", "--host", "0.0.0.0", "--port", "5173"]
