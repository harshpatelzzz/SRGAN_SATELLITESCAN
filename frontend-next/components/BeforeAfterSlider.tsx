'use client'

import { useState, useRef, useEffect } from 'react'
import { motion } from 'framer-motion'

interface BeforeAfterSliderProps {
  beforeImage: string
  afterImage: string
}

export default function BeforeAfterSlider({ beforeImage, afterImage }: BeforeAfterSliderProps) {
  const [sliderPosition, setSliderPosition] = useState(50)
  const [isDragging, setIsDragging] = useState(false)
  const containerRef = useRef<HTMLDivElement>(null)

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging || !containerRef.current) return
    
    const rect = containerRef.current.getBoundingClientRect()
    const x = e.clientX - rect.left
    const percentage = (x / rect.width) * 100
    setSliderPosition(Math.max(0, Math.min(100, percentage)))
  }

  const handleMouseDown = () => setIsDragging(true)
  const handleMouseUp = () => setIsDragging(false)

  useEffect(() => {
    if (isDragging) {
      const handleGlobalMouseMove = (e: MouseEvent) => {
        if (!containerRef.current) return
        const rect = containerRef.current.getBoundingClientRect()
        const x = e.clientX - rect.left
        const percentage = (x / rect.width) * 100
        setSliderPosition(Math.max(0, Math.min(100, percentage)))
      }

      const handleGlobalMouseUp = () => setIsDragging(false)

      window.addEventListener('mousemove', handleGlobalMouseMove)
      window.addEventListener('mouseup', handleGlobalMouseUp)

      return () => {
        window.removeEventListener('mousemove', handleGlobalMouseMove)
        window.removeEventListener('mouseup', handleGlobalMouseUp)
      }
    }
  }, [isDragging])

  return (
    <div className="relative rounded-xl overflow-hidden border border-dark-border">
      <div
        ref={containerRef}
        className="relative w-full h-64 md:h-96 cursor-col-resize"
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseUp}
      >
        {/* Before Image (Background) */}
        <div className="absolute inset-0">
          <img
            src={beforeImage}
            alt="Before"
            className="w-full h-full object-contain"
          />
          <div className="absolute top-2 left-2 px-3 py-1 bg-dark-card/80 backdrop-blur-sm rounded-lg text-sm font-semibold">
            Original
          </div>
        </div>

        {/* After Image (Clipped) */}
        <div
          className="absolute inset-0 overflow-hidden"
          style={{ clipPath: `inset(0 ${100 - sliderPosition}% 0 0)` }}
        >
          <img
            src={afterImage}
            alt="After"
            className="w-full h-full object-contain"
          />
          <div className="absolute top-2 right-2 px-3 py-1 bg-indigo-500/80 backdrop-blur-sm rounded-lg text-sm font-semibold">
            Upscaled (4×)
          </div>
        </div>

        {/* Slider Line */}
        <div
          className="absolute top-0 bottom-0 w-1 bg-white shadow-lg z-10"
          style={{ left: `${sliderPosition}%` }}
        >
          {/* Slider Handle */}
          <motion.div
            className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-10 h-10 bg-white rounded-full shadow-lg flex items-center justify-center cursor-grab active:cursor-grabbing"
            onMouseDown={handleMouseDown}
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
          >
            <div className="flex gap-1">
              <div className="w-1 h-4 bg-dark-border rounded-full" />
              <div className="w-1 h-4 bg-dark-border rounded-full" />
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  )
}
