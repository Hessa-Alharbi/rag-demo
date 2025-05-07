#!/bin/bash

# تنظيف ذاكرة التخزين المؤقت قبل بدء التشغيل
echo "Cleaning up cache directories..."
rm -rf /tmp/*
rm -rf ~/.cache/pip/*

# تعيين متغيرات البيئة لتحسين استخدام الذاكرة
export PYTORCH_NO_CUDA_MEMORY_CACHING=1
export MALLOC_ARENA_MAX=2
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TRANSFORMERS_OFFLINE=1 # منع تنزيل النماذج في وقت التشغيل

# تم تحديد الحد الأقصى من ذاكرة Python
export PYTHONMALLOC=malloc
export MALLOC_TRIM_THRESHOLD_=65536
export MPLBACKEND=Agg # تقليل استخدام الذاكرة إذا كان يستخدم matplotlib

# تم تغيير إعدادات بدء التشغيل لـ uvicorn
echo "Starting application with memory optimizations..."
exec python -m uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1 --limit-concurrency 4 --timeout-keep-alive 30 --log-level info 
