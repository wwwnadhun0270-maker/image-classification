@echo off
echo ================================================
echo   Installing CUDA PyTorch for GTX 3050
echo ================================================

"C:/Users/Nadhu/AppData/Local/Programs/Python/Python311/python.exe" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --upgrade

echo.
echo Verifying GPU...
"C:/Users/Nadhu/AppData/Local/Programs/Python/Python311/python.exe" -c "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT FOUND'); print('VRAM:', round(torch.cuda.get_device_properties(0).total_memory/1e9,1), 'GB') if torch.cuda.is_available() else None"

echo.
echo Installing other packages...
"C:/Users/Nadhu/AppData/Local/Programs/Python/Python311/python.exe" -m pip install ultralytics opencv-python pyyaml

echo.
echo All done! Now run coco_to_yolo.py then train.py
pause
