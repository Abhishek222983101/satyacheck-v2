import type { NextConfig } from "next";

// /v1/* proxies to the inference server (localhost:8000 in dev; set API_PROXY for prod).
// Same-origin media/SSE/dossier links then just work, and CORS stops mattering.
const API = process.env.API_PROXY ?? "http://localhost:8000";

const nextConfig: NextConfig = {
  turbopack: { root: __dirname },
  async rewrites() {
    return [{ source: "/v1/:path*", destination: `${API}/v1/:path*` }];
  },
};

export default nextConfig;
