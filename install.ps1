[CmdletBinding()]
param(
    [string]$Version = "latest",
    [switch]$Dev,
    [switch]$Upgrade,
    [switch]$Check,
    [switch]$DryRun,
    [switch]$SkipSandbox,
    [switch]$SkipMemoryModels,
    [string]$ModelBundle,
    [switch]$NonInteractive,
    [switch]$Uninstall,
    [switch]$PurgeData,
    [switch]$Yes
)

$ErrorActionPreference = "Stop"
$UvVersion = "0.11.28"
$Repository = "https://github.com/jony-del/AgentwithLLM"
# Filled by tools/build_release_assets.py; published scripts always pin one release.
$ReleaseTag = ""
$TemporaryRoot = $null

if ($Uninstall -and ($Dev -or $Upgrade -or $Check -or $SkipSandbox -or $SkipMemoryModels -or $ModelBundle)) {
    [Console]::Error.WriteLine("[usage] -Uninstall cannot be combined with install/upgrade options")
    exit 2
}
if ($SkipMemoryModels -and $ModelBundle) {
    [Console]::Error.WriteLine("[usage] -SkipMemoryModels cannot be combined with -ModelBundle")
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
    if ($PSScriptRoot -and (Test-Path -LiteralPath (Join-Path $PSScriptRoot "installer\install.py"))) {
        if ($Uninstall -and -not (Test-Path -LiteralPath (Join-Path $PSScriptRoot "agent_core\uninstall.py"))) {
            throw "Source checkout is missing agent_core/uninstall.py"
        }
        return (Resolve-Path -LiteralPath $PSScriptRoot).Path
    }

    if ($Dev) { throw "-Dev requires a persistent source checkout" }
    $tag = if ($Version -eq "latest") { $ReleaseTag } else { $Version }
    if ($tag -notmatch '^v[0-9]+\.[0-9]+\.[0-9]+([-.][0-9A-Za-z.-]+)?$') {
        throw "Use an installer from a published GitHub Release, or specify -Version vX.Y.Z"
    }

    $script:TemporaryRoot = Join-Path ([IO.Path]::GetTempPath()) ("polaris-install-" + [guid]::NewGuid())
    New-Item -ItemType Directory -Path $script:TemporaryRoot | Out-Null
    $base = "$Repository/releases/download/$tag"
    $archive = Join-Path $script:TemporaryRoot "polaris-installer.zip"
    $sums = Join-Path $script:TemporaryRoot "SHA256SUMS"
    Write-Host "Downloading Polaris $tag release..."
    Invoke-WebRequest -UseBasicParsing "$base/polaris-installer.zip" -OutFile $archive
    Invoke-WebRequest -UseBasicParsing "$base/SHA256SUMS" -OutFile $sums

    $entries = @(Get-Content -LiteralPath $sums | Where-Object { $_ -match '^[0-9a-fA-F]{64}\s+\*?polaris-installer\.zip$' })
    if ($entries.Count -ne 1) { throw "SHA256SUMS must contain exactly one polaris-installer.zip entry" }
    $entry = $entries[0]
    $expected = ($entry -split "\s+")[0].ToLowerInvariant()
    $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $archive).Hash.ToLowerInvariant()
    if ($actual -ne $expected) { throw "SHA-256 mismatch for polaris-installer.zip" }

    $source = Join-Path $script:TemporaryRoot "source"
    Expand-Archive -LiteralPath $archive -DestinationPath $source
    if (-not (Test-Path -LiteralPath (Join-Path $source "installer\install.py"))) {
        throw "Release archive is missing installer/install.py"
    }
    if (-not (Test-Path -LiteralPath (Join-Path $source "agent_core\uninstall.py"))) {
        throw "Release archive is missing agent_core/uninstall.py"
    }
    $release = Get-Content -LiteralPath (Join-Path $source "release.json") -Raw | ConvertFrom-Json
    if ($release.tag -ne $tag) { throw "Release bundle version does not match requested $tag" }
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
    # uv's installer may only update future shells. The Python worker needs it now.
    $env:PATH = (Split-Path -Parent $uv) + [IO.Path]::PathSeparator + $env:PATH
    if (-not ($Uninstall -or $Check -or $DryRun)) {
        & $uv python install 3.12
        if ($LASTEXITCODE -ne 0) { throw "uv could not install Python 3.12" }
    } else {
        $env:UV_PYTHON_DOWNLOADS = "never"
    }
    # Never bootstrap uninstall/check from the project's .venv.  uv normally
    # prefers a discovered project environment, but that environment may be the
    # exact target the uninstall worker must remove.
    $pythonOutput = & $uv python find --system --no-project 3.12 | Select-Object -Last 1
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
        if ($SkipMemoryModels) { $arguments += "--skip-memory-models" }
        if ($ModelBundle) { $arguments += @("--model-bundle", $ModelBundle) }
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
        $resolvedTemp = (Resolve-Path -LiteralPath $TemporaryRoot).Path
        $tempBase = [IO.Path]::GetFullPath([IO.Path]::GetTempPath())
        if ($resolvedTemp.StartsWith($tempBase, [StringComparison]::OrdinalIgnoreCase) -and
            [IO.Path]::GetFileName($resolvedTemp) -match '^polaris-install-[0-9a-f-]{36}$') {
            Remove-Item -LiteralPath $resolvedTemp -Recurse -Force
        }
    }
}
exit $exitCode
