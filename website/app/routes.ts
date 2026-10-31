import { index, type RouteConfig, route } from "@react-router/dev/routes";

export default [
  route("api/search", "routes/search.ts"),
  route("llms.txt", "llms/index.ts"),
  route("llms-full.txt", "llms/full.ts"),
  route("llms.mdx/*", "llms/mdx.ts"),
  index("routes/docs.tsx", { id: "introduction" }),
  route("*", "routes/docs.tsx"),
] satisfies RouteConfig;
