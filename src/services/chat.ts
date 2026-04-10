import { api } from "./api";

export async function sendChatMessage(message: string, history: object[]) {
  return api.post("/api/chat", { message, history });
}