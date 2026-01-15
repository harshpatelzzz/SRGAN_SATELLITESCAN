'use client'

import { useState, useRef } from 'react'
import { motion } from 'framer-motion'
import { Upload, Zap, BarChart3, Cpu, Sparkles, ArrowRight } from 'lucide-react'
import ImageUpload from '@/components/ImageUpload'
import BeforeAfterSlider from '@/components/BeforeAfterSlider'
import StatsCard from '@/components/StatsCard'

export default function Home() {
  const [uploadedImage, setUploadedImage] = useState<string | null>(null)
  const [upscaledImage, setUpscaledImage] = useState<string | null>(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const [metrics, setMetrics] = useState<any>(null)

  const handleUpscale = async (imageFile: File) => {
    setIsProcessing(true)
    const formData = new FormData()
    formData.append('file', imageFile)

    try {
      const response = await fetch('/api/upscale', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }))
        throw new Error(errorData.detail || `Server error: ${response.status}`)
      }

      const data = await response.json()
      
      if (!data.success) {
        throw new Error(data.error || 'Upscaling failed')
      }
      
      setUpscaledImage(data.upscaled_image_base64)
      setMetrics({
        originalSize: data.original_size,
        upscaledSize: data.upscaled_size,
        processingTime: data.processing_time,
      })
    } catch (error: any) {
      console.error('Error:', error)
      const errorMessage = error.message || 'Failed to upscale image. Please check that the API server is running and try again.'
      alert(errorMessage)
    } finally {
      setIsProcessing(false)
    }
  }

  return (
    <div className="min-h-screen bg-dark-bg">
      {/* Hero Section */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="relative overflow-hidden"
      >
        <div className="absolute inset-0 bg-gradient-to-br from-indigo-900/20 via-purple-900/20 to-pink-900/20" />
        <div className="relative container mx-auto px-4 py-20">
          <div className="text-center max-w-4xl mx-auto">
            <motion.div
              initial={{ scale: 0.9 }}
              animate={{ scale: 1 }}
              transition={{ duration: 0.5 }}
              className="inline-block mb-6"
            >
              <Sparkles className="w-16 h-16 text-indigo-400" />
            </motion.div>
            <h1 className="text-6xl font-bold mb-6 gradient-text">
              SRGAN Super-Resolution
            </h1>
            <p className="text-xl text-dark-textMuted mb-8">
              AI-Powered 4× Upscaling for Satellite Imagery
            </p>
            <p className="text-lg text-dark-textMuted mb-12">
              Transform low-resolution satellite images into high-resolution using
              advanced Generative Adversarial Networks
            </p>
          </div>
        </div>
      </motion.div>

      {/* Main Content */}
      <div className="container mx-auto px-4 py-12">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-12">
          {/* Upload Section */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="glass-effect rounded-2xl p-8"
          >
            <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
              <Upload className="w-6 h-6 text-indigo-400" />
              Upload Image
            </h2>
            <ImageUpload
              onImageSelect={(file) => {
                const reader = new FileReader()
                reader.onload = (e) => setUploadedImage(e.target?.result as string)
                reader.readAsDataURL(file)
                handleUpscale(file)
              }}
              isProcessing={isProcessing}
            />
          </motion.div>

          {/* Results Section */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.4 }}
            className="glass-effect rounded-2xl p-8"
          >
            <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
              <Zap className="w-6 h-6 text-purple-400" />
              Results
            </h2>
            {isProcessing ? (
              <div className="flex flex-col items-center justify-center h-64">
                <motion.div
                  animate={{ rotate: 360 }}
                  transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                  className="w-16 h-16 border-4 border-indigo-400 border-t-transparent rounded-full"
                />
                <p className="mt-4 text-dark-textMuted">Processing image...</p>
              </div>
            ) : upscaledImage ? (
              <div className="space-y-4">
                <BeforeAfterSlider
                  beforeImage={uploadedImage!}
                  afterImage={upscaledImage}
                />
                {metrics && (
                  <div className="grid grid-cols-2 gap-4 mt-4">
                    <StatsCard
                      label="Original"
                      value={metrics.originalSize}
                      icon={<Cpu className="w-4 h-4" />}
                    />
                    <StatsCard
                      label="Upscaled"
                      value={metrics.upscaledSize}
                      icon={<Zap className="w-4 h-4" />}
                    />
                  </div>
                )}
              </div>
            ) : (
              <div className="flex items-center justify-center h-64 text-dark-textMuted">
                Upload an image to see results
              </div>
            )}
          </motion.div>
        </div>

        {/* Features Grid */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.6 }}
          className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12"
        >
          <div className="glass-effect rounded-xl p-6 card-hover">
            <BarChart3 className="w-8 h-8 text-indigo-400 mb-4" />
            <h3 className="text-xl font-bold mb-2">Advanced Metrics</h3>
            <p className="text-dark-textMuted">
              PSNR and SSIM evaluation for quality assessment
            </p>
          </div>
          <div className="glass-effect rounded-xl p-6 card-hover">
            <Cpu className="w-8 h-8 text-purple-400 mb-4" />
            <h3 className="text-xl font-bold mb-2">Deep Learning</h3>
            <p className="text-dark-textMuted">
              16 residual blocks with VGG19 perceptual loss
            </p>
          </div>
          <div className="glass-effect rounded-xl p-6 card-hover">
            <Sparkles className="w-8 h-8 text-pink-400 mb-4" />
            <h3 className="text-xl font-bold mb-2">4× Enhancement</h3>
            <p className="text-dark-textMuted">
              Transform 64×64 to 256×256 with AI precision
            </p>
          </div>
        </motion.div>

        {/* CTA Section */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.6, delay: 0.8 }}
          className="text-center"
        >
          <a
            href="/architecture"
            className="inline-flex items-center gap-2 px-8 py-4 bg-gradient-to-r from-indigo-500 to-purple-500 rounded-xl font-semibold hover:from-indigo-600 hover:to-purple-600 transition-all"
          >
            View Architecture
            <ArrowRight className="w-5 h-5" />
          </a>
        </motion.div>
      </div>
    </div>
  )
}
