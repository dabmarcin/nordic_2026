@echo off
title Nordic 2026 - Invest Dashboard
cd /d "%~dp0"
echo.
echo ========================================
echo  Nordic 2026 - Invest Dashboard
echo ========================================
echo.
echo Starting invest_app.py on port 8501...
echo.
echo Open in browser: http://localhost:8501
echo.
streamlit run invest_app.py --logger.level=error
pause
