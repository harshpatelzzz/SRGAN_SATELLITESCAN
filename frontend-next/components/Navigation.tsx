'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { Home, Layers, BarChart3, Database } from 'lucide-react'
import { motion } from 'framer-motion'

export default function Navigation() {
  const pathname = usePathname()

  const navItems = [
    { href: '/', label: 'Home', icon: Home },
    { href: '/architecture', label: 'Architecture', icon: Layers },
    { href: '/metrics', label: 'Metrics', icon: BarChart3 },
    { href: '/dataset', label: 'Dataset', icon: Database },
  ]

  return (
    <nav className="glass-effect border-b border-dark-border">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          <Link href="/" className="text-2xl font-bold gradient-text">
            SRGAN
          </Link>
          <div className="flex gap-1">
            {navItems.map((item) => {
              const Icon = item.icon
              const isActive = pathname === item.href
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={`relative px-4 py-2 rounded-lg transition-colors ${
                    isActive
                      ? 'text-indigo-400'
                      : 'text-dark-textMuted hover:text-dark-text'
                  }`}
                >
                  <div className="flex items-center gap-2">
                    <Icon className="w-4 h-4" />
                    <span className="hidden md:inline">{item.label}</span>
                  </div>
                  {isActive && (
                    <motion.div
                      layoutId="activeTab"
                      className="absolute inset-0 bg-indigo-500/10 rounded-lg -z-10"
                      transition={{ type: 'spring', bounce: 0.2, duration: 0.6 }}
                    />
                  )}
                </Link>
              )
            })}
          </div>
        </div>
      </div>
    </nav>
  )
}
