const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export const api = {
  post: async (path: string, body: object) => {
    const res = await fetch(`${BASE_URL}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    return res.json();
  },
  get: async (path: string) => {
    const res = await fetch(`${BASE_URL}${path}`);
    return res.json();
  },
};