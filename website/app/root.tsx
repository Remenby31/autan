import "./app.css";
import { i18nProvider } from "fumadocs-ui/i18n";
import { RootProvider } from "fumadocs-ui/provider/react-router";
import {
  isRouteErrorResponse,
  Links,
  Meta,
  Outlet,
  Scripts,
  ScrollRestoration,
  useLocation,
  useNavigate,
} from "react-router";
import StaticSearchDialog from "@/components/search";
import { localePath } from "@/content/paths";
import { translations } from "@/lib/layout.shared";
import { parsePath } from "@/lib/locale";
import favicon from "../../docs/assets/favicon.svg?url";
import type { Route } from "./+types/root";

export const links: Route.LinksFunction = () => [
  { rel: "icon", href: favicon, type: "image/svg+xml" },
];

import NotFound from "./routes/not-found";

export function Layout({ children }: { children: React.ReactNode }) {
  const { pathname, hash } = useLocation();
  const navigate = useNavigate();
  const { locale, slug } = parsePath(pathname);
  const i18n = {
    ...i18nProvider(translations, locale),
    onLocaleChange: (next: string) =>
      navigate(localePath(slug, next as typeof locale) + hash),
  };
  return (
    <html lang={locale} suppressHydrationWarning>
      <head>
        <meta charSet="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <Meta />
        <Links />
      </head>
      <body className="flex flex-col min-h-screen">
        <RootProvider
          i18n={i18n}
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
