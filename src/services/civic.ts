import { api } from "./api";

export async function get311Estimate(requestType: string, neighborhood: string) {
  return api.get(`/api/311/predict?type=${requestType}&neighborhood=${neighborhood}`);
}