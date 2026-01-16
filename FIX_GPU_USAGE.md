# 🔧 Fix GPU Usage - RTX 4060 Not Being Used

## Problem
Task Manager shows:
- **RTX 4060**: 0% usage
- **Intel UHD Graphics**: 100% usage

This means Windows is routing graphics operations to the integrated GPU instead of the dedicated RTX 4060.

## ✅ Code Fixes Applied
1. ✅ Set `DEVICE = 'cuda:0'` explicitly
2. ✅ Added `torch.cuda.set_device(0)`
3. ✅ Set `CUDA_VISIBLE_DEVICES=0` environment variable
4. ✅ Added GPU memory allocation test

## 🔧 Windows Graphics Settings Fix (REQUIRED)

### Method 1: Windows Graphics Settings (Recommended)

1. **Open Windows Settings**
   - Press `Win + I`
   - Or: Start Menu > Settings

2. **Navigate to Graphics Settings**
   - Go to: **System** > **Display**
   - Scroll down to **Graphics**

3. **Add Python to Graphics Settings**
   - Click **"Browse"**
   - Navigate to Python executable:
     - Usually: `C:\Python313\python.exe`
     - Or: `C:\Users\YourName\AppData\Local\Programs\Python\Python313\python.exe`
     - Or find it: `where python` in PowerShell

4. **Set to High Performance**
   - After adding Python.exe, click **"Options"**
   - Select **"High Performance"** (this uses RTX 4060)
   - Click **"Save"**

5. **Restart Training**
   - Close any running Python processes
   - Restart training: `python main.py pretrain`

### Method 2: NVIDIA Control Panel

1. **Open NVIDIA Control Panel**
   - Right-click desktop
   - Select **"NVIDIA Control Panel"**

2. **Manage 3D Settings**
   - Go to: **Manage 3D Settings** > **Program Settings**

3. **Add Python**
   - Click **"Add"**
   - Browse to: `python.exe` (same path as above)
   - Or select from dropdown if already used

4. **Set Preferred Graphics Processor**
   - Find **"Preferred graphics processor"**
   - Select **"High-performance NVIDIA processor"**
   - Click **"Apply"**

5. **Restart Training**

### Method 3: Environment Variable (Temporary)

Run training with:
```powershell
$env:CUDA_VISIBLE_DEVICES="0"
$env:CUDA_DEVICE_ORDER="PCI_BUS_ID"
python main.py pretrain
```

## ✅ Verify GPU Usage

### Check with nvidia-smi
```powershell
nvidia-smi
```
Should show:
- Process using GPU
- GPU utilization > 0%
- Memory usage

### Check Task Manager
1. Open Task Manager (`Ctrl + Shift + Esc`)
2. Go to **Performance** tab
3. Select **GPU** (should show RTX 4060)
4. Look for:
   - **GPU usage** > 0%
   - **Dedicated GPU memory** being used

### Check Training Log
Look for:
```
Using device: cuda:0 (NVIDIA GeForce RTX 4060 Laptop GPU)
GPU Memory: X.X GB
```

## 🎯 Expected Results

After fixing:
- **RTX 4060**: Should show 50-100% usage during training
- **Intel UHD**: Should be 0-10% (just for display)
- **Training speed**: 10-50x faster than CPU

## ⚠️ Common Issues

1. **Python path not found**
   - Find Python: `where python` in PowerShell
   - Use full path in Graphics Settings

2. **Changes not taking effect**
   - Restart Python process
   - Restart computer (if needed)

3. **Still using Intel UHD**
   - Make sure you selected "High Performance" (not "Power Saving")
   - Check both Windows Settings AND NVIDIA Control Panel

## 📝 Quick Checklist

- [ ] Added Python.exe to Windows Graphics Settings
- [ ] Set to "High Performance" (RTX 4060)
- [ ] Restarted training
- [ ] Checked nvidia-smi shows GPU usage
- [ ] Task Manager shows RTX 4060 usage > 0%

---

**Status**: Code is fixed. Windows graphics routing needs to be configured.
