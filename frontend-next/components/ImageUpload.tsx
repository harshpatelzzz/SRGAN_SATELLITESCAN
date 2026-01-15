'use client'

import { useState, useRef } from 'react'
import { motion } from 'framer-motion'
import { Upload, X } from 'lucide-react'

interface ImageUploadProps {
  onImageSelect: (file: File) => void
  isProcessing: boolean
}

export default function ImageUpload({ onImageSelect, isProcessing }: ImageUploadProps) {
  const [dragActive, setDragActive] = useState(false)
  const [preview, setPreview] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true)
    } else if (e.type === 'dragleave') {
      setDragActive(false)
    }
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0])
    }
  }

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0])
    }
  }

  const handleFile = (file: File) => {
    if (!file.type.startsWith('image/')) {
      alert('Please select an image file')
      return
    }

    const reader = new FileReader()
    reader.onload = (e) => {
      setPreview(e.target?.result as string)
    }
    reader.readAsDataURL(file)
    onImageSelect(file)
  }

  const clearPreview = () => {
    setPreview(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  return (
    <div className="space-y-4">
      <div
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        className={`
          relative border-2 border-dashed rounded-xl p-8 text-center transition-all
          ${dragActive ? 'border-indigo-400 bg-indigo-400/10' : 'border-dark-border'}
          ${preview ? 'border-solid' : ''}
        `}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={handleChange}
          className="hidden"
          disabled={isProcessing}
        />

        {preview ? (
          <div className="relative">
            <img
              src={preview}
              alt="Preview"
              className="max-w-full max-h-64 mx-auto rounded-lg"
            />
            <button
              onClick={clearPreview}
              className="absolute top-2 right-2 p-2 bg-dark-card rounded-full hover:bg-dark-border transition-colors"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        ) : (
          <div className="space-y-4">
            <motion.div
              animate={{ y: [0, -10, 0] }}
              transition={{ duration: 2, repeat: Infinity }}
            >
              <Upload className="w-12 h-12 mx-auto text-dark-textMuted" />
            </motion.div>
            <div>
              <p className="text-lg font-semibold mb-2">
                Drag & drop an image here
              </p>
              <p className="text-sm text-dark-textMuted mb-4">
                or click to browse
              </p>
              <button
                onClick={() => fileInputRef.current?.click()}
                disabled={isProcessing}
                className="px-6 py-2 bg-indigo-500 hover:bg-indigo-600 rounded-lg transition-colors disabled:opacity-50"
              >
                Select Image
              </button>
            </div>
            <p className="text-xs text-dark-textMuted">
              Supports: PNG, JPG, JPEG, BMP, TIF (Max 16MB)
            </p>
          </div>
        )}
      </div>
    </div>
  )
}
