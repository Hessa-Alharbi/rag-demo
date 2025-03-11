#!/bin/bash

# Colors for better visibility
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting NavidRAG components...${NC}"

# Check if Redis is running
echo -e "${YELLOW}Checking Redis...${NC}"
if command -v redis-cli &> /dev/null; then
    if redis-cli ping > /dev/null 2>&1; then
        echo -e "${GREEN}Redis is running${NC}"
    else
        echo -e "${RED}Redis is not running. Starting Redis...${NC}"
        redis-server &
        sleep 2
        if redis-cli ping > /dev/null 2>&1; then
            echo -e "${GREEN}Redis started successfully${NC}"
        else
            echo -e "${RED}Failed to start Redis. Please start it manually${NC}"
            exit 1
        fi
    fi
else
    echo -e "${RED}Redis CLI not found. Please install Redis or make sure it's in your PATH${NC}"
    exit 1
fi

# Start Celery worker
echo -e "${YELLOW}Starting Celery worker...${NC}"
celery -A celery_app worker --loglevel=info &
CELERY_PID=$!

# Give Celery a moment to start up
sleep 3

# Start FastAPI application
echo -e "${YELLOW}Starting FastAPI application...${NC}"
uvicorn main:app --reload --host 0.0.0.0 --port 8000 &
API_PID=$!

echo -e "${GREEN}All components started:${NC}"
echo -e "  - ${YELLOW}Redis: Running${NC}"
echo -e "  - ${YELLOW}Celery worker: PID $CELERY_PID${NC}"
echo -e "  - ${YELLOW}FastAPI: PID $API_PID${NC}"
echo -e "${GREEN}API is available at http://localhost:8000${NC}"
echo -e "${GREEN}API documentation at http://localhost:8000/docs${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop all services${NC}"

# Wait for Ctrl+C
trap "kill $CELERY_PID $API_PID; echo -e '${RED}Shutting down...${NC}'; exit" INT
wait
