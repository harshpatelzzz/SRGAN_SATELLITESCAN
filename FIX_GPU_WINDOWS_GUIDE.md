# 🔧 Fix GPU Usage - Step-by-Step Windows Guide

## Problem
RTX 4060 shows 0% usage, Intel UHD at 100%

## ✅ Method 1: NVIDIA Control Panel (EASIEST - Recommended)

### Steps:
1. **Open NVIDIA Control Panel**
   - Right-click on desktop (empty area)
   - Select **"NVIDIA Control Panel"**
   - (If not visible, search "NVIDIA Control Panel" in Start Menu)

2. **Navigate to 3D Settings**
   - Left sidebar: Click **"Manage 3D Settings"**
   - Click **"Program Settings"** tab

3. **Add Python**
   - Click **"Add"** button (top of the list)
   - Click **"Browse"** button
   - Navigate to: `C:\Python313\python.exe`
   - Click **"Open"**

4. **Set Graphics Processor**
   - Find **"Preferred graphics processor"** in the list
   - Change from "Use global setting" to:
     - **"High-performance NVIDIA processor"** ✅
   - Click **"Apply"** (bottom right)

5. **Restart Training**
   - Close any running Python processes
   - Restart: `python main.py pretrain`

---

## ✅ Method 2: Windows Graphics Settings (Windows 11)

### Steps:
1. **Open Settings**
   - Press `Win + I`
   - Or: Start Menu > Settings

2. **Navigate to Graphics**
   - Click **"System"** (left sidebar)
   - Click **"Display"**
   - Scroll down to find **"Graphics"** section
   - Click **"Graphics settings"** (or "Graphics" link)

3. **Add Desktop App**
   - Under **"Desktop app"** section
   - Click **"Browse"** button
   - Navigate to: `C:\Python313\python.exe`
   - Click **"Add"**

4. **Set Performance**
   - After adding, Python.exe should appear in the list
   - Click on **"python.exe"**
   - Click **"Options"** button
   - Select **"High performance"** (RTX 4060)
   - Click **"Save"**

5. **Restart Training**

---

## ✅ Method 3: Windows Graphics Settings (Windows 10)

### Steps:
1. **Open Settings**
   - Press `Win + I`
   - Go to: **System** > **Display**

2. **Graphics Settings**
   - Scroll down to **"Graphics settings"**
   - Click it

3. **Classic App**
   - Select **"Classic app"** (not Desktop app)
   - Click **"Browse"**
   - Find: `C:\Python313\python.exe`
   - Click **"Add"**

4. **Set Options**
   - Click on **python.exe** in the list
   - Click **"Options"**
   - Select **"High performance"**
   - **Save**

---

## ✅ Method 4: Direct Environment Variable (Quick Test)

Run training with explicit GPU selection:

```powershell
# Set environment variables
$env:CUDA_VISIBLE_DEVICES="0"
$env:CUDA_DEVICE_ORDER="PCI_BUS_ID"

# Start training
cd Z:\AIMLLABEL
python main.py pretrain
```

---

## ✅ Method 5: Create GPU-Only Batch File

Create `train_gpu.bat`:

```batch
@echo off
set CUDA_VISIBLE_DEVICES=0
set CUDA_DEVICE_ORDER=PCI_BUS_ID
cd /d Z:\AIMLLABEL
python main.py pretrain
```

Then run: `train_gpu.bat`

---

## 🔍 Verify GPU Usage

### Check with nvidia-smi:
```powershell
nvidia-smi
```

Should show:
- Python process in the list
- GPU utilization > 0%
- Memory being used

### Check Task Manager:
1. Open Task Manager (`Ctrl + Shift + Esc`)
2. **Performance** tab
3. Select **GPU** (RTX 4060)
4. Should show:
   - GPU usage > 0%
   - Dedicated GPU memory > 0 MB

---

## 🎯 Which Method to Use?

1. **NVIDIA Control Panel** - Easiest, most reliable ✅
2. **Windows Graphics Settings** - Good if NVIDIA Control Panel not available
3. **Environment Variables** - Quick test, but may not persist
4. **Batch File** - Convenient for repeated use

---

## ⚠️ Troubleshooting

### If "Browse" button not found:
- Try **NVIDIA Control Panel** method instead
- Or use **Method 4** (Environment Variables)

### If changes don't work:
1. Restart computer
2. Make sure you selected "High Performance" (not "Power Saving")
3. Check both Windows Settings AND NVIDIA Control Panel

### If still using Intel UHD:
- Verify Python path is correct: `C:\Python313\python.exe`
- Try adding `pythonw.exe` as well
- Check if there are multiple Python installations

---

## 📝 Quick Checklist

- [ ] Added Python.exe to NVIDIA Control Panel OR Windows Graphics Settings
- [ ] Set to "High Performance" / "High-performance NVIDIA processor"
- [ ] Applied/Saved changes
- [ ] Restarted training
- [ ] Verified with nvidia-smi or Task Manager

---

**Recommended**: Use **NVIDIA Control Panel** method - it's the most reliable!
