#!/bin/bash

# Script to initialize Git repository for the IMDB ABSA project

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Initializing Git repository for IMDB ABSA project...${NC}"

# Initialize Git repository
git init

# Add all files to the repository
git add .

# Initial commit
git commit -m "Initial commit: IMDB ABSA project structure"

# Instructions for GitHub setup
echo -e "${YELLOW}Repository initialized locally.${NC}"
echo -e "${YELLOW}To push to GitHub, create a repository on GitHub and run:${NC}"
echo -e "git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git"
echo -e "git branch -M main"
echo -e "git push -u origin main"

echo -e "${GREEN}Done!${NC}" 