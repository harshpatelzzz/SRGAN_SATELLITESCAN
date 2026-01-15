import type { Metadata } from 'next'
import './globals.css'
import Navigation from '@/components/Navigation'

export const metadata: Metadata = {
  title: 'SRGAN - Satellite Imagery Super-Resolution',
  description: 'AI-powered 4× upscaling of satellite imagery using Super-Resolution Generative Adversarial Networks',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className="dark">
      <body>
        <Navigation />
        {children}
      </body>
    </html>
  )
}
