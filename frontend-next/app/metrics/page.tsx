'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { ArrowLeft, BarChart3, TrendingUp, Award } from 'lucide-react'
import Link from 'next/link'

export default function MetricsPage() {
  const [metrics, setMetrics] = useState<any>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetch('/api/metrics')
      .then(res => res.json())
      .then(data => {
        setMetrics(data)
        setLoading(false)
      })
      .catch(() => setLoading(false))
  }, [])

  return (
    <div className="min-h-screen bg-dark-bg">
      <div className="container mx-auto px-4 py-12">
        <Link
          href="/"
          className="inline-flex items-center gap-2 text-dark-textMuted hover:text-dark-text mb-8 transition-colors"
        >
          <ArrowLeft className="w-4 h-4" />
          Back to Home
        </Link>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="space-y-8"
        >
          <div className="text-center">
            <h1 className="text-5xl font-bold mb-4 gradient-text">
              Performance Metrics
            </h1>
            <p className="text-xl text-dark-textMuted">
              Quantitative evaluation of SRGAN performance
            </p>
          </div>

          {loading ? (
            <div className="text-center py-12">
              <div className="inline-block w-12 h-12 border-4 border-indigo-400 border-t-transparent rounded-full animate-spin" />
            </div>
          ) : metrics && metrics.srgan_psnr ? (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* PSNR */}
              <motion.div
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                className="glass-effect rounded-2xl p-8"
              >
                <div className="flex items-center gap-3 mb-6">
                  <BarChart3 className="w-8 h-8 text-indigo-400" />
                  <h2 className="text-2xl font-bold">PSNR (dB)</h2>
                </div>
                <div className="space-y-4">
                  <div className="flex justify-between items-center p-4 bg-dark-card rounded-lg">
                    <span>SRGAN</span>
                    <span className="text-2xl font-bold text-indigo-400">
                      {metrics.srgan_psnr.toFixed(2)}
                    </span>
                  </div>
                  <div className="flex justify-between items-center p-4 bg-dark-card rounded-lg">
                    <span>Bicubic</span>
                    <span className="text-2xl font-bold text-dark-textMuted">
                      {metrics.bicubic_psnr.toFixed(2)}
                    </span>
                  </div>
                  <div className="flex justify-between items-center p-4 bg-indigo-500/10 rounded-lg border border-indigo-500/20">
                    <span>Improvement</span>
                    <span className={`text-xl font-bold ${
                      metrics.srgan_psnr > metrics.bicubic_psnr ? 'text-green-400' : 'text-red-400'
                    }`}>
                      {(metrics.srgan_psnr - metrics.bicubic_psnr).toFixed(2)} dB
                    </span>
                  </div>
                </div>
              </motion.div>

              {/* SSIM */}
              <motion.div
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                className="glass-effect rounded-2xl p-8"
              >
                <div className="flex items-center gap-3 mb-6">
                  <TrendingUp className="w-8 h-8 text-purple-400" />
                  <h2 className="text-2xl font-bold">SSIM</h2>
                </div>
                <div className="space-y-4">
                  <div className="flex justify-between items-center p-4 bg-dark-card rounded-lg">
                    <span>SRGAN</span>
                    <span className="text-2xl font-bold text-purple-400">
                      {metrics.srgan_ssim.toFixed(4)}
                    </span>
                  </div>
                  <div className="flex justify-between items-center p-4 bg-dark-card rounded-lg">
                    <span>Bicubic</span>
                    <span className="text-2xl font-bold text-dark-textMuted">
                      {metrics.bicubic_ssim.toFixed(4)}
                    </span>
                  </div>
                  <div className="flex justify-between items-center p-4 bg-purple-500/10 rounded-lg border border-purple-500/20">
                    <span>Improvement</span>
                    <span className={`text-xl font-bold ${
                      metrics.srgan_ssim > metrics.bicubic_ssim ? 'text-green-400' : 'text-red-400'
                    }`}>
                      {(metrics.srgan_ssim - metrics.bicubic_ssim).toFixed(4)}
                    </span>
                  </div>
                </div>
              </motion.div>
            </div>
          ) : (
            <div className="glass-effect rounded-2xl p-12 text-center">
              <Award className="w-16 h-16 mx-auto text-dark-textMuted mb-4" />
              <p className="text-lg text-dark-textMuted">
                No evaluation results available yet.
              </p>
              <p className="text-sm text-dark-textMuted mt-2">
                Run evaluation to see metrics: <code className="bg-dark-card px-2 py-1 rounded">python main.py evaluate</code>
              </p>
            </div>
          )}
        </motion.div>
      </div>
    </div>
  )
}
