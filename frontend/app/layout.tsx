import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'PredictiveCare',
  description: 'Intelligent Patient Health Forecasting',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  // suppressHydrationWarning covers only the attributes of <html> and <body> themselves
  // (it is one level deep). Browser extensions such as Grammarly, Dark Reader and
  // password managers write attributes onto these two elements before React hydrates,
  // which React reports as a hydration mismatch. Everything inside still hydrates strictly.
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={inter.className} suppressHydrationWarning>{children}</body>
    </html>
  );
}
