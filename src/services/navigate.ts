import { api } from "./api";

export async function getRoute(from: string, to: string, safetyWeight: number) {
  return api.post("/api/navigate", { from, to, safety_weight: safetyWeight });
}