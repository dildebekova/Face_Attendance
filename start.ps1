param([switch]$DownloadResearchModels)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
function RunChecked([string]$Program, [string[]]$Arguments) {
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Ошибка команды: $Program $Arguments" }
}
if (!(Get-Command docker -ErrorAction SilentlyContinue)) { throw 'Установите Docker Desktop и запустите его.' }
RunChecked 'docker' @('info')
if (!(Test-Path '.env')) { RunChecked 'python' @('init_config.py') }
if (!(Test-Path 'models/sha256.json')) {
    if (!$DownloadResearchModels) { throw 'Прочитайте MODEL_LICENSES.md. Для исследовательской загрузки повторите: .\start.ps1 -DownloadResearchModels' }
    RunChecked 'python' @('setup_models.py','--accept-research-licenses')
}
RunChecked 'docker' @('compose','up','--build','-d')
Write-Host 'Откройте http://localhost:8080. Ключ входа — ADMIN_KEY из .env.'
Write-Host 'Проверка моделей: docker compose exec backend python smoke_models.py'
Start-Process 'http://localhost:8080'
