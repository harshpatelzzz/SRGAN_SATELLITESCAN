'use client'

import { motion } from 'framer-motion'
import { ArrowLeft, Database, Download, Satellite, Image as ImageIcon } from 'lucide-react'
import Link from 'next/link'

export default function DatasetPage() {
  const degradationSteps = [
    { step: 1, name: 'Gaussian Blur', desc: 'σ = 1.2, simulates atmospheric scattering' },
    { step: 2, name: 'Bicubic Downsampling', desc: '256×256 → 64×64 (4× reduction)' },
    { step: 3, name: 'Gaussian Noise', desc: 'σ = 0.01, simulates sensor noise' },
  ]

  const datasets = [
    {
      name: 'DIV2K',
      desc: 'Standard benchmark for super-resolution',
      images: '900 (800 train + 100 val)',
      size: '~7GB',
      quality: 'Very High',
      recommended: true,
    },
    {
      name: 'UC Merced',
      desc: 'Satellite-specific land use dataset',
      images: '2,100',
      size: '~200MB',
      quality: 'High',
      recommended: false,
    },
    {
      name: 'Sentinel-2',
      desc: 'Real satellite imagery (requires download)',
      images: 'Variable',
      size: 'Large',
      quality: 'Very High',
      recommended: false,
    },
  ]

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
          className="space-y-12"
        >
          <div className="text-center">
            <h1 className="text-5xl font-bold mb-4 gradient-text">
              Dataset & Degradation Pipeline
            </h1>
            <p className="text-xl text-dark-textMuted">
              How we create LR-HR pairs for training
            </p>
          </div>

          {/* Degradation Pipeline */}
          <div className="glass-effect rounded-2xl p-8">
            <div className="flex items-center gap-3 mb-6">
              <ImageIcon className="w-8 h-8 text-indigo-400" />
              <h2 className="text-3xl font-bold">Degradation Pipeline</h2>
            </div>
            <p className="text-dark-textMuted mb-8">
              Realistic simulation of satellite image degradation factors
            </p>
            <div className="space-y-4">
              {degradationSteps.map((step, idx) => (
                <motion.div
                  key={idx}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: idx * 0.1 }}
                  className="flex items-start gap-4 p-6 bg-dark-card rounded-lg border border-dark-border"
                >
                  <div className="flex-shrink-0 w-12 h-12 bg-indigo-500/20 rounded-full flex items-center justify-center font-bold text-indigo-400">
                    {step.step}
                  </div>
                  <div className="flex-1">
                    <h3 className="text-xl font-bold mb-2">{step.name}</h3>
                    <p className="text-dark-textMuted">{step.desc}</p>
                  </div>
                  {idx < degradationSteps.length - 1 && (
                    <div className="absolute left-1/2 transform -translate-x-1/2 mt-16 w-0.5 h-8 bg-indigo-500/30" />
                  )}
                </motion.div>
              ))}
            </div>
            <div className="mt-6 p-4 bg-indigo-500/10 rounded-lg border border-indigo-500/20">
              <p className="text-sm">
                <strong>Result:</strong> HR (256×256) → LR (64×64) pairs for training
              </p>
            </div>
          </div>

          {/* Available Datasets */}
          <div className="glass-effect rounded-2xl p-8">
            <div className="flex items-center gap-3 mb-6">
              <Database className="w-8 h-8 text-purple-400" />
              <h2 className="text-3xl font-bold">Available Datasets</h2>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {datasets.map((dataset, idx) => (
                <motion.div
                  key={idx}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: idx * 0.1 }}
                  className={`p-6 rounded-xl border ${
                    dataset.recommended
                      ? 'bg-indigo-500/10 border-indigo-500/30'
                      : 'bg-dark-card border-dark-border'
                  }`}
                >
                  {dataset.recommended && (
                    <div className="inline-block px-3 py-1 bg-indigo-500/20 text-indigo-400 text-xs font-semibold rounded-full mb-3">
                      Recommended
                    </div>
                  )}
                  <div className="flex items-center gap-2 mb-3">
                    <Satellite className="w-5 h-5 text-purple-400" />
                    <h3 className="text-xl font-bold">{dataset.name}</h3>
                  </div>
                  <p className="text-dark-textMuted text-sm mb-4">{dataset.desc}</p>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-dark-textMuted">Images:</span>
                      <span className="font-semibold">{dataset.images}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-dark-textMuted">Size:</span>
                      <span className="font-semibold">{dataset.size}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-dark-textMuted">Quality:</span>
                      <span className="font-semibold">{dataset.quality}</span>
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          </div>

          {/* Download Instructions */}
          <div className="glass-effect rounded-2xl p-8">
            <div className="flex items-center gap-3 mb-6">
              <Download className="w-8 h-8 text-pink-400" />
              <h2 className="text-3xl font-bold">Download Dataset</h2>
            </div>
            <div className="space-y-4">
              <div className="bg-dark-card rounded-lg p-6 border border-dark-border">
                <h3 className="text-lg font-bold mb-3">Using Command Line</h3>
                <code className="block bg-dark-bg p-4 rounded-lg text-sm">
                  python main.py download-dataset --dataset div2k --auto-config
                </code>
              </div>
              <div className="bg-dark-card rounded-lg p-6 border border-dark-border">
                <h3 className="text-lg font-bold mb-3">Manual Download</h3>
                <ul className="list-disc list-inside space-y-2 text-dark-textMuted">
                  <li>DIV2K: <a href="https://data.vision.ee.ethz.ch/cvl/DIV2K/" className="text-indigo-400 hover:underline">data.vision.ee.ethz.ch/cvl/DIV2K/</a></li>
                  <li>UC Merced: <a href="http://weecology.org/data/ucmerced/" className="text-indigo-400 hover:underline">weecology.org/data/ucmerced/</a></li>
                </ul>
              </div>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  )
}
