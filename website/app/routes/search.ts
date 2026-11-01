import { createFromSource } from "fumadocs-core/search/server";
import { source } from "@/lib/source";

// fumadocs' default multilingual tokenizer indexes English and French pages alike.
export const loader = createFromSource(source).staticGET;
