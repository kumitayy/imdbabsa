@echo off
echo Initializing Git repository for IMDB ABSA project...

REM Initialize Git repository
git init

REM Add all files to the repository
git add .

REM Initial commit
git commit -m "Initial commit: IMDB ABSA project structure"

REM Instructions for GitHub setup
echo.
echo Repository initialized locally.
echo To push to GitHub, create a repository on GitHub and run:
echo git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
echo git branch -M main
echo git push -u origin main
echo.
echo Done!
pause 