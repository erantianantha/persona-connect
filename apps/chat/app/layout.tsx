import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Anantha Datta Eranti — Chat with Me",
  description:
    "Chat directly with Anantha Datta Eranti's AI twin. Ask about his projects, skills, background, or book a time to connect. Powered by RAG over his resume and GitHub.",
  openGraph: {
    title: "Anantha Datta Eranti — Chat with Me",
    description: "CS student at Scaler building AI voice agents and full-stack systems. Chat with Anantha's AI twin directly.",
    type: "website",
  },
};


export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
      <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap"
          rel="stylesheet"
        />
      </head>
      <body>{children}</body>
    </html>
  );
}
