# Launch the Model4 teacher fine-tune queue as a detached process (waits for CryptBB v3 + idle GPU first).
# Usage: powershell -ExecutionPolicy Bypass -File model4/launch_teachers.ps1
$root = Split-Path $PSScriptRoot -Parent
. (Join-Path $root 'environment.ps1')
$m4 = Join-Path $root 'model4'
$env:PYTHONPATH = ((Join-Path $m4 'src'), (Join-Path $m4 'shims'), (Join-Path $m4 'site'), $env:PYTHONPATH) -join ';'
$env:PYTHONUNBUFFERED = '1'
$run = Join-Path $m4 'runs/darkbay_AB_teachers_20261114'
New-Item -ItemType Directory -Force $run | Out-Null
$p = Start-Process -FilePath $thesisPython -ArgumentList '-B', '-u', '-m', 'model4_teachers.gpu_queue' `
    -WorkingDirectory $root -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $run 'queue.stdout.txt') -RedirectStandardError (Join-Path $run 'queue.stderr.txt')
@{ pid = $p.Id; launched_at = (Get-Date).ToUniversalTime().ToString('o'); python = $thesisPython } |
    ConvertTo-Json | Set-Content -Encoding utf8 (Join-Path $run 'queue_launch.json')
Write-Output "queue pid $($p.Id)"
