<#
.SYNOPSIS
    Tamil Music Composer Agent launcher.
.DESCRIPTION
    Activates the ace-step-1.5 virtual environment and runs the Tamil Composer
    agent with the given song specification JSON file.
.PARAMETER SpecFile
    Path to the song specification JSON file.
.EXAMPLE
    .\agents\tamil_composer\compose.ps1 agents\tamil_composer\examples\kanavaa.json
#>
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$SpecFile
)

$ErrorActionPreference = 'Stop'

# Resolve ace-step-1.5 project root (two levels up from this script).
$ProjectRoot = (Resolve-Path "$PSScriptRoot\..\..")

# Activate the venv.
& "$ProjectRoot\venv\Scripts\Activate.ps1"

# Run the agent from the project root so relative imports work.
Push-Location $ProjectRoot
try {
    python -m agents.tamil_composer $SpecFile
}
finally {
    Pop-Location
}
