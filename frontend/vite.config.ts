import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // proxy API requests to backend if needed
      "/chat": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
});