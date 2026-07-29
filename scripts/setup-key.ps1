# Copies GROQ_API_KEY from an existing project into RoleFit's local .env and
# into Vercel, then redeploys and health-checks.
#
# The key is read from disk and handed to Vercel directly. It is never printed,
# never logged, and never leaves this machine except to Vercel itself.
#
#   powershell -ExecutionPolicy Bypass -File scripts\setup-key.ps1

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot

$sources = @(
  'C:\Users\Work_Pari\job-outreach-tool\.env.local',
  'C:\Users\Work_Pari\deal-flow-mvp\.env.local'
)

$key = $null
$from = $null
foreach ($s in $sources) {
  if (-not (Test-Path $s)) { continue }
  $line = Get-Content $s | Where-Object { $_ -match '^\s*GROQ_API_KEY\s*=' } | Select-Object -First 1
  if ($line) {
    $key = ($line -split '=', 2)[1].Trim().Trim('"').Trim("'")
    $from = $s
    break
  }
}

if (-not $key) {
  Write-Output "No GROQ_API_KEY found in any known project .env.local."
  Write-Output "Checked:"
  $sources | ForEach-Object { Write-Output "  $_" }
  exit 1
}

Write-Output "Found GROQ_API_KEY in $from (length $($key.Length)). Value not shown."

# 1. Local .env, so the evals can run on this machine.
$envFile = Join-Path $repo '.env'
$existing = @()
if (Test-Path $envFile) {
  $existing = Get-Content $envFile | Where-Object { $_ -notmatch '^\s*ROLEFIT_GROQ_API_KEY\s*=' }
}
$out = $existing + "ROLEFIT_GROQ_API_KEY=$key"
Set-Content -Path $envFile -Value $out -Encoding utf8
Write-Output "Wrote ROLEFIT_GROQ_API_KEY to $envFile"

# 2. Vercel. Remove any stale value first, because `env add` refuses to
#    overwrite and would otherwise fail the whole script.
Push-Location $repo
try {
  & vercel env rm ROLEFIT_GROQ_API_KEY production --yes 2>$null | Out-Null

  $tmp = [System.IO.Path]::GetTempFileName()
  try {
    Set-Content -Path $tmp -Value $key -NoNewline -Encoding ascii
    Get-Content $tmp | & vercel env add ROLEFIT_GROQ_API_KEY production
  } finally {
    Remove-Item $tmp -Force -ErrorAction SilentlyContinue
  }
  Write-Output "Pushed ROLEFIT_GROQ_API_KEY to Vercel production"

  # 3. Redeploy so the new variable is actually in the running function.
  Write-Output "Redeploying..."
  & vercel --prod --yes | Out-Null
} finally {
  Pop-Location
}

# 4. Prove it worked.
Write-Output ""
Write-Output "Health check:"
try {
  $h = Invoke-RestMethod -Uri 'https://rolefit-wine.vercel.app/api/health' -TimeoutSec 90
  $h | ConvertTo-Json -Compress | Write-Output
  if ($h.llm_key_present -and $h.status -eq 'ok') {
    Write-Output ""
    Write-Output "RoleFit is fully live. Run:  python -m evals.run_eval"
  } else {
    Write-Output ""
    Write-Output "Still degraded. Check the values above."
  }
} catch {
  Write-Output "Health check failed: $_"
}
