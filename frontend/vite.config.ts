import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// In development the page runs on :5173 and /api calls are passed to the FastAPI server on :8000.
// In production FastAPI serves the built files itself (frontend/dist), so both share one address.
export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/api": "http://localhost:8000" } },
});
