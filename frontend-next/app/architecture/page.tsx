'use client'

import { motion } from 'framer-motion'
import { ArrowLeft, Layers, Zap, Brain, Target } from 'lucide-react'
import Link from 'next/link'

export default function ArchitecturePage() {
  const generatorBlocks = [
    { name: 'Input Conv', desc: '9×9 kernel, 64 features' },
    { name: 'Residual Block 1-16', desc: 'Conv → BN → PReLU → Conv → BN' },
    { name: 'Post-Residual', desc: 'Conv + BatchNorm' },
    { name: 'Upsample Block 1', desc: 'PixelShuffle 2×' },
    { name: 'Upsample Block 2', desc: 'PixelShuffle 2×' },
    { name: 'Output Conv', desc: '9×9 kernel, RGB output' },
  ]

  const discriminatorBlocks = [
    { name: 'Input Conv', desc: '3×3 kernel, 64 features' },
    { name: 'Block 1', desc: '64 → 64, stride 2' },
    { name: 'Block 2', desc: '64 → 128' },
    { name: 'Block 3', desc: '128 → 128, stride 2' },
    { name: 'Block 4', desc: '128 → 256' },
    { name: 'Block 5', desc: '256 → 256, stride 2' },
    { name: 'Block 6', desc: '256 → 512' },
    { name: 'Block 7', desc: '512 → 512, stride 2' },
    { name: 'Global Pooling', desc: 'AdaptiveAvgPool2d' },
    { name: 'FC Layers', desc: '512 → 1024 → 1' },
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
              Model Architecture
            </h1>
            <p className="text-xl text-dark-textMuted">
              Deep learning architecture for satellite imagery super-resolution
            </p>
          </div>

          {/* Generator */}
          <div className="glass-effect rounded-2xl p-8">
            <div className="flex items-center gap-3 mb-6">
              <Layers className="w-8 h-8 text-indigo-400" />
              <h2 className="text-3xl font-bold">Generator (SRResNet-based)</h2>
            </div>
            <p className="text-dark-textMuted mb-8">
              Transforms low-resolution (64×64) images to high-resolution (256×256) using
              16 residual blocks and PixelShuffle upsampling.
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {generatorBlocks.map((block, idx) => (
                <motion.div
                  key={idx}
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: idx * 0.1 }}
                  className="bg-dark-card rounded-lg p-4 border border-dark-border"
                >
                  <h3 className="font-semibold mb-2">{block.name}</h3>
                  <p className="text-sm text-dark-textMuted">{block.desc}</p>
                </motion.div>
              ))}
            </div>
            <div className="mt-6 p-4 bg-indigo-500/10 rounded-lg border border-indigo-500/20">
              <p className="text-sm">
                <strong>Parameters:</strong> 1,546,774 | <strong>Scale Factor:</strong> 4×
              </p>
            </div>
          </div>

          {/* Discriminator */}
          <div className="glass-effect rounded-2xl p-8">
            <div className="flex items-center gap-3 mb-6">
              <Target className="w-8 h-8 text-purple-400" />
              <h2 className="text-3xl font-bold">Discriminator</h2>
            </div>
            <p className="text-dark-textMuted mb-8">
              CNN classifier that distinguishes real HR images from generated ones,
              enabling adversarial training.
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {discriminatorBlocks.map((block, idx) => (
                <motion.div
                  key={idx}
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: idx * 0.1 }}
                  className="bg-dark-card rounded-lg p-4 border border-dark-border"
                >
                  <h3 className="font-semibold mb-2">{block.name}</h3>
                  <p className="text-sm text-dark-textMuted">{block.desc}</p>
                </motion.div>
              ))}
            </div>
            <div className="mt-6 p-4 bg-purple-500/10 rounded-lg border border-purple-500/20">
              <p className="text-sm">
                <strong>Parameters:</strong> 5,213,505 | <strong>Output:</strong> Probability (0-1)
              </p>
            </div>
          </div>

          {/* Loss Functions */}
          <div className="glass-effect rounded-2xl p-8">
            <div className="flex items-center gap-3 mb-6">
              <Brain className="w-8 h-8 text-pink-400" />
              <h2 className="text-3xl font-bold">Loss Functions</h2>
            </div>
            <div className="space-y-4">
              <div className="bg-dark-card rounded-lg p-6 border border-dark-border">
                <h3 className="text-xl font-bold mb-2 flex items-center gap-2">
                  <Zap className="w-5 h-5 text-indigo-400" />
                  Total Loss: L = L_VGG + 10⁻³ × L_GAN + L_MSE
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-4">
                  <div className="p-4 bg-indigo-500/10 rounded-lg">
                    <p className="font-semibold mb-1">VGG Perceptual Loss</p>
                    <p className="text-sm text-dark-textMuted">Weight: 1.0</p>
                    <p className="text-xs text-dark-textMuted mt-2">
                      Feature-level similarity using VGG19 (relu5_4)
                    </p>
                  </div>
                  <div className="p-4 bg-purple-500/10 rounded-lg">
                    <p className="font-semibold mb-1">Adversarial Loss</p>
                    <p className="text-sm text-dark-textMuted">Weight: 10⁻³</p>
                    <p className="text-xs text-dark-textMuted mt-2">
                      Binary cross-entropy from discriminator
                    </p>
                  </div>
                  <div className="p-4 bg-pink-500/10 rounded-lg">
                    <p className="font-semibold mb-1">MSE Pixel Loss</p>
                    <p className="text-sm text-dark-textMuted">Weight: 1.0</p>
                    <p className="text-xs text-dark-textMuted mt-2">
                      Pixel-wise mean squared error
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  )
}
