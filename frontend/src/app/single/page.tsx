import { redirect } from "next/navigation";

export default function SinglePage() {
  redirect("/analyze?mode=single");
}
