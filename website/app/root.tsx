import "./app.css";
import { RootProvider } from "fumadocs-ui/provider/react-router";
import {
  isRouteErrorResponse,
  Links,
  Meta,
  Outlet,
  Scripts,
  ScrollRestoration,
} from "react-router";
import StaticSearchDialog from "@/components/search";
import type { Route } from "./+types/root";
import NotFound from "./routes/not-found";

export function Layout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <meta charSet="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <Meta />
        <Links />
      </head>
      <body className="flex flex-col min-h-screen">
        <RootProvider
          search={{ SearchDialog: StaticSearchDialog }}
          theme={{ attribute: ["class", "data-theme"] }}
        >
          {children}
        </RootProvider>
        <ScrollRestoration />
        <Scripts />
      </body>
    </html>
  );
}

export default function App() {
  return <Outlet />;
}

export function ErrorBoundary({ error }: Route.ErrorBoundaryProps) {
  if (isRouteErrorResponse(error) && error.status === 404) return <NotFound />;
  const details =
    error instanceof Error ? error.message : "An unexpected error occurred.";
  return (
    <main className="pt-16 p-4 w-full max-w-[1400px] mx-auto">
      <h1>Error</h1>
      <p>{details}</p>
    </main>
  );
}
