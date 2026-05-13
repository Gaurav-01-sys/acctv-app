#!/bin/bash
LOG=/tmp/startup.log

echo "=== Startup $(date) ===" > $LOG
echo "Python: $(python --version 2>&1)" >> $LOG
echo "" >> $LOG

echo "=== Disk space ===" >> $LOG
df -h / >> $LOG 2>&1
echo "" >> $LOG

echo "=== Pre-installed packages ===" >> $LOG
pip list 2>/dev/null >> $LOG
echo "" >> $LOG

echo "=== Installing requirements ===" >> $LOG
pip install --no-cache-dir -r requirements.txt >> $LOG 2>&1
echo "pip exit: $?" >> $LOG
echo "" >> $LOG

echo "=== Post-install verification ===" >> $LOG
python -c "import cv2; print(f'cv2: {cv2.__version__}')" >> $LOG 2>&1
python -c "import numpy; print(f'numpy: {numpy.__version__}')" >> $LOG 2>&1
python -c "import sklearn; print(f'sklearn: {sklearn.__version__}')" >> $LOG 2>&1
python -c "import torch; print(f'torch: {torch.__version__}')" >> $LOG 2>&1
python -c "import ultralytics; print(f'ultralytics: {ultralytics.__version__}')" >> $LOG 2>&1
echo "" >> $LOG

echo "=== Starting app ===" >> $LOG
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
