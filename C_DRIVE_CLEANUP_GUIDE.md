# C Drive Cleanup Guide

## Current Status
- **C Drive Free Space**: 1.58 GB (0.6% free) ⚠️ CRITICAL
- **Python Packages**: 4.59 GB (in `C:\Users\harsh\AppData\Roaming\Python`)
- **Temp Files**: Cleaned (freed 1.34 GB)

## What's Taking Space

### 1. Python Packages (4.59 GB) - CANNOT DELETE
Location: `C:\Users\harsh\AppData\Roaming\Python\Python313\site-packages`

**Why it's there**: Packages installed with `pip install --user` go here.

**Options**:
- ✅ **Keep it** (needed for your project)
- ⚠️ **Move to Z drive** (complex, may break things)
- ✅ **Use virtual environment** (for future projects)

### 2. PyTorch Models Cache
Location: `C:\Users\harsh\.cache\torch\`

**Can clean**: Yes (will re-download when needed)
- VGG19 model: ~500 MB
- Other cached models

### 3. Other Large Folders to Check

Run these commands to find large folders:

```powershell
# Find large folders in your user directory
Get-ChildItem C:\Users\harsh -Directory | ForEach-Object {
    $size = (Get-ChildItem $_.FullName -Recurse -ErrorAction SilentlyContinue | 
             Measure-Object -Property Length -Sum).Sum / 1GB
    if ($size -gt 1) {
        [PSCustomObject]@{Folder=$_.Name; SizeGB=[math]::Round($size, 2)}
    }
} | Sort-Object SizeGB -Descending
```

## Immediate Actions

### ✅ Already Done
- Cleaned temp files: **Freed 1.34 GB**

### 🔧 Recommended Actions

1. **Clean PyTorch Cache** (if not training right now):
   ```powershell
   Remove-Item -Recurse -Force "$env:USERPROFILE\.cache\torch" -ErrorAction SilentlyContinue
   ```
   This will free ~500 MB (models re-download when needed)

2. **Run Windows Disk Cleanup**:
   - Press `Win + R`, type `cleanmgr`, press Enter
   - Select C drive
   - Clean system files

3. **Check Downloads Folder**:
   ```powershell
   Get-ChildItem "$env:USERPROFILE\Downloads" | Measure-Object -Property Length -Sum
   ```

4. **Check Browser Cache**:
   - Chrome: `C:\Users\harsh\AppData\Local\Google\Chrome\User Data\Default\Cache`
   - Can be several GB

## Long-term Solutions

1. **Move Project to Z Drive** (already done ✅)
2. **Use Virtual Environment** for future Python projects:
   ```powershell
   python -m venv Z:\AIMLLABEL\venv
   .\venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
   This keeps packages on Z drive, not C drive.

3. **Uninstall Unused Programs**:
   - Settings → Apps → Uninstall unused programs

4. **Move User Folders to Z Drive**:
   - Downloads, Documents, Desktop can be moved to Z drive

## Current Free Space: 1.58 GB

⚠️ **WARNING**: Your C drive is critically low on space. Consider:
- Moving more files to Z drive
- Uninstalling unused programs
- Using Windows Disk Cleanup tool
