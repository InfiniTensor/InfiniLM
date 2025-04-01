@echo off
setlocal

:: Check if project path is provided
if "%~0"=="" (
    echo Usage: install.bat PROJECT_PATH [XMAKE_CONFIG_FLAGS]
    exit /b 1
)

:: Set INFINI_ROOT
set "INFINI_ROOT=%USERPROFILE%\.infini"

:: Check if INFINI_ROOT\bin is already in PATH, if not, add it
echo %PATH% | findstr /I /C:"%INFINI_ROOT%\bin" >nul
if %errorlevel% neq 0 set "PATH=%INFINI_ROOT%\bin;%PATH%"

:: Convert relative path to absolute path
for %%I in ("%~1") do set ABS_PATH=%%~fI

:: Change to the project directory
cd %ABS_PATH%

:: Build xmake config flags
set XMAKE_FLAGS=
set i=0
for %%A in (%*) do (
    if !i! gtr  set XMAKE_FLAGS=!XMAKE_FLAGS! %%A
    set /a i+=1
)

:: Start installation
xmake clean -a
if %errorlevel% neq 0 exit /b %errorlevel%

xmake f %XMAKE_FLAGS% -cv
if %errorlevel% neq 0 exit /b %errorlevel%

xmake
if %errorlevel% neq 0 exit /b %errorlevel%

xmake install
if %errorlevel% neq 0 exit /b %errorlevel%

xmake build infiniop-test
if %errorlevel% neq 0 exit /b %errorlevel%

xmake install infiniop-test
if %errorlevel% neq 0 exit /b %errorlevel%
