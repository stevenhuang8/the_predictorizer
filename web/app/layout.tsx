import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import Link from "next/link";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: {
    template: "%s | Eco Prediction",
    default: "Eco Prediction",
  },
  description:
    "Point-in-time forecasts of US inflation, unemployment and FOMC decisions, " +
    "scored against what actually happened.",
};

const NAV = [
  { href: "/", label: "Overview" },
  { href: "/forecasts", label: "Forecasts" },
  { href: "/track-record", label: "Track record" },
] as const;

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="flex min-h-full flex-col">
        <header className="border-b border-line">
          <nav className="mx-auto flex max-w-4xl flex-wrap items-center gap-x-6 gap-y-2 px-4 py-4">
            <Link href="/" className="font-semibold tracking-tight">
              Eco Prediction
            </Link>
            <ul className="flex flex-wrap gap-x-5 gap-y-1 text-sm text-muted">
              {NAV.map(({ href, label }) => (
                <li key={href}>
                  <Link href={href} className="hover:text-foreground">
                    {label}
                  </Link>
                </li>
              ))}
            </ul>
          </nav>
        </header>
        <main className="mx-auto w-full max-w-4xl flex-1 px-4 py-10">{children}</main>
        <footer className="border-t border-line">
          <p className="mx-auto max-w-4xl px-4 py-4 text-xs text-muted">
            Data: FRED / ALFRED vintages, as published at the time of each forecast.
          </p>
        </footer>
      </body>
    </html>
  );
}
