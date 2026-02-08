import { redirect } from "next/navigation";

export default function BatchPage() {
  redirect("/analyze?mode=batch");
}
