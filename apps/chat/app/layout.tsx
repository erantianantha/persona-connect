import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Anantha Datta Eranti — AI Representative",
  description:
    "Chat with Anantha Datta Eranti's AI representative. Ask about his background, projects, AI Calling Agent with RAG, or book an interview directly.",
  openGraph: {
    title: "Anantha Datta Eranti — AI Representative",
    description: "AI persona of Anantha Datta Eranti — CS student at Scaler School of Technology building AI voice agents.",
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
