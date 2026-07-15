[CmdletBinding()]
param(
    [string]$Version = "latest",
    [switch]$Dev,
    [switch]$Upgrade,
    [switch]$Check,
    [switch]$DryRun,
    [switch]$SkipSandbox,
    [switch]$NonInteractive,
    [switch]$Uninstall,
    [switch]$PurgeData,
    [switch]$Yes
)

$ErrorActionPreference = "Stop"
$UvVersion = "0.11.28"
$Repository = "https://github.com/jony-del/AgentwithLLM"
$TemporaryRoot = $null

if ($Uninstall -and ($Dev -or $Upgrade -or $Check -or $SkipSandbox)) {
    [Console]::Error.WriteLine("[usage] -Uninstall cannot be combined with -Dev, -Upgrade, -Check, or -SkipSandbox")
    exit 2
}
if ((-not $Uninstall) -and ($PurgeData -or $Yes)) {
    [Console]::Error.WriteLine("[usage] -PurgeData and -Yes require -Uninstall")
    exit 2
}
if ($Uninstall -and $NonInteractive -and (-not $Yes) -and (-not $DryRun)) {
    [Console]::Error.WriteLine("[usage] non-interactive uninstall requires -Yes (or use -DryRun)")
    exit 2
}

function Get-VerifiedSource {
    $localInstaller = Join-Path $PSScriptRoot "installer\install.py"
    if ($PSScriptRoot -and (Test-Path -LiteralPath $localInstaller)) {
        if ($Uninstall -and -not (Test-Path -LiteralPath (Join-Path $PSScriptRoot "agent_core\uninstall.py"))) {
            throw "Source checkout is missing agent_core/uninstall.py"
        }
        return (Resolve-Path -LiteralPath $PSScriptRoot).Path
    }

    $script:TemporaryRoot = Join-Path ([IO.Path]::GetTempPath()) ("polaris-install-" + [guid]::NewGuid())
    New-Item -ItemType Directory -Path $script:TemporaryRoot | Out-Null
    $base = if ($Version -eq "latest") {
        "$Repository/releases/latest/download"
    } else {
        "$Repository/releases/download/$Version"
    }
    $archive = Join-Path $script:TemporaryRoot "polaris-source.zip"
    $sums = Join-Path $script:TemporaryRoot "SHA256SUMS"
    Write-Host "Downloading Polaris $Version release..."
    Invoke-WebRequest -UseBasicParsing "$base/polaris-source.zip" -OutFile $archive
    Invoke-WebRequest -UseBasicParsing "$base/SHA256SUMS" -OutFile $sums

    $entry = Get-Content -LiteralPath $sums | Where-Object { $_ -match "\s\*?polaris-source\.zip$" } | Select-Object -First 1
    if (-not $entry) { throw "SHA256SUMS does not contain polaris-source.zip" }
    $expected = ($entry -split "\s+")[0].ToLowerInvariant()
    $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $archive).Hash.ToLowerInvariant()
    if ($actual -ne $expected) { throw "SHA-256 mismatch for polaris-source.zip" }

    $source = Join-Path $script:TemporaryRoot "source"
    Expand-Archive -LiteralPath $archive -DestinationPath $source
    if (-not (Test-Path -LiteralPath (Join-Path $source "installer\install.py"))) {
        throw "Release archive is missing installer/install.py"
    }
    if (-not (Test-Path -LiteralPath (Join-Path $source "agent_core\uninstall.py"))) {
        throw "Release archive is missing agent_core/uninstall.py"
    }
    return $source
}

function Get-UvCommand {
    $command = Get-Command uv -ErrorAction SilentlyContinue
    if ($command) { return $command.Source }
    if ($Uninstall -or $Check -or $DryRun) {
        throw "uv is missing; uninstall/check/dry-run mode will not install it"
    }
    Write-Host "Installing uv $UvVersion..."
    Invoke-RestMethod "https://astral.sh/uv/$UvVersion/install.ps1" | Invoke-Expression
    $candidates = @(
        (Join-Path $env:USERPROFILE ".local\bin\uv.exe"),
        (Join-Path $env:LOCALAPPDATA "Programs\uv\uv.exe")
    )
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) { return $candidate }
    }
    $command = Get-Command uv -ErrorAction SilentlyContinue
    if (-not $command) { throw "uv installation completed but uv.exe was not found" }
    return $command.Source
}

$exitCode = 10
try {
    $source = Get-VerifiedSource
    if ($Dev -and $TemporaryRoot) {
        throw "-Dev requires a persistent source checkout; run this script from the repository"
    }
    $uv = Get-UvCommand
    if (-not ($Uninstall -or $Check -or $DryRun)) {
        & $uv python install 3.12
        if ($LASTEXITCODE -ne 0) { throw "uv could not install Python 3.12" }
    } else {
        $env:UV_PYTHON_DOWNLOADS = "never"
    }
    $pythonOutput = & $uv python find 3.12 | Select-Object -Last 1
    $findExitCode = $LASTEXITCODE
    $python = if ($pythonOutput) { $pythonOutput.Trim() } else { "" }
    if ($findExitCode -ne 0 -or -not $python -or -not (Test-Path -LiteralPath $python)) {
        throw "Python 3.12 is unavailable; uninstall/check/dry-run mode will not install it"
    }

    if ($Uninstall) {
        $arguments = @()
        if ($PurgeData) { $arguments += "--purge-data" }
        if ($Yes) { $arguments += "--yes" }
        if ($DryRun) { $arguments += "--dry-run" }
        if ($NonInteractive) { $arguments += "--non-interactive" }
        & $python (Join-Path $source "agent_core\uninstall.py") @arguments
        $exitCode = $LASTEXITCODE
    } else {
        $arguments = @("--source", $source)
        if ($Dev) { $arguments += "--dev" }
        if ($Upgrade) { $arguments += "--upgrade" }
        if ($Check) { $arguments += "--check" }
        if ($DryRun) { $arguments += "--dry-run" }
        if ($SkipSandbox) { $arguments += "--skip-sandbox" }
        if ($NonInteractive) { $arguments += "--non-interactive" }
        & $python (Join-Path $source "installer\install.py") @arguments
        $exitCode = $LASTEXITCODE
    }
}
catch {
    [Console]::Error.WriteLine("[error] " + $_.Exception.Message)
    $exitCode = 10
}
finally {
    if ($TemporaryRoot -and (Test-Path -LiteralPath $TemporaryRoot)) {
        Remove-Item -LiteralPath $TemporaryRoot -Recurse -Force
    }
}
exit $exitCode
