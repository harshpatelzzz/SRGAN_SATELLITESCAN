'use client'

import { ReactNode } from 'react'

interface StatsCardProps {
  label: string
  value: string
  icon: ReactNode
}

export default function StatsCard({ label, value, icon }: StatsCardProps) {
  return (
    <div className="glass-effect rounded-lg p-4">
      <div className="flex items-center gap-2 mb-2">
        <div className="text-indigo-400">{icon}</div>
        <span className="text-sm text-dark-textMuted">{label}</span>
      </div>
      <p className="text-xl font-bold">{value}</p>
    </div>
  )
}
