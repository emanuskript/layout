import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";

import { Header } from "@/components/layout/Header";
import { JotaiProvider } from "@/lib/context/JotaiProvider";
import { ThemeProvider } from "@/lib/context/ThemeProvider";

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
  title: "Manuscript Layout Analysis",
  description: "Detect and annotate medieval manuscript layout elements using YOLO models",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={`${geistSans.variable} ${geistMono.variable} antialiased`}>
        <ThemeProvider>
          <JotaiProvider>
            <Header />
            <main className="min-h-[calc(100vh-57px)]">{children}</main>
          </JotaiProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}
