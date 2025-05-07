#!/usr/bin/env bash

# تحسين استخدام الذاكرة
export PYTORCH_NO_CUDA_MEMORY_CACHING=1
export MALLOC_ARENA_MAX=2
export PYTHONUNBUFFERED=1

# ضبط إعدادات uvicorn لتحسين استخدام الذاكرة
export UVICORN_WORKERS=1
export UVICORN_BACKLOG=8
export UVICORN_TIMEOUT=120 