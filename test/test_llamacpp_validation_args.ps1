$ErrorActionPreference = "Stop"

. "$PSScriptRoot/../.github/scripts/llamacpp_validation_args.ps1"

function Assert-Arguments {
    param(
        [string[]] $Actual,
        [string[]] $Expected,
        [string] $Case
    )

    if ($Actual.Count -ne $Expected.Count) {
        throw "$Case returned $($Actual.Count) arguments; expected $($Expected.Count)."
    }

    for ($index = 0; $index -lt $Expected.Count; $index++) {
        if ($Actual[$index] -ne $Expected[$index]) {
            throw "$Case argument $index was '$($Actual[$index])'; expected '$($Expected[$index])'."
        }
    }
}

Assert-Arguments `
    -Actual @(Get-LlamaCppValidationModelArguments -Models "" -Lite $false) `
    -Expected @() `
    -Case "default mode"

Assert-Arguments `
    -Actual @(
        Get-LlamaCppValidationModelArguments `
            -Models "" `
            -Lite $false `
            -CapabilityProfile "k2-horizon-v1" `
            -CapabilityModels "K2-Horizon-0.9B-GGUF"
    ) `
    -Expected @(
        "--capability-profile",
        "k2-horizon-v1",
        "--capability-model",
        "K2-Horizon-0.9B-GGUF"
    ) `
    -Case "hot selection capability profile"

Assert-Arguments `
    -Actual @(
        Get-LlamaCppValidationModelArguments `
            -Models "K2-Horizon-0.9B-GGUF,K2-Horizon-3.7B-GGUF" `
            -Lite $false `
            -CapabilityProfile "k2-horizon-v1" `
            -CapabilityModels "K2-Horizon-0.9B-GGUF"
    ) `
    -Expected @(
        "--model",
        "K2-Horizon-0.9B-GGUF",
        "--model",
        "K2-Horizon-3.7B-GGUF",
        "--capability-profile",
        "k2-horizon-v1",
        "--capability-model",
        "K2-Horizon-0.9B-GGUF"
    ) `
    -Case "capability profile"

Assert-Arguments `
    -Actual @(
        Get-LlamaCppValidationModelArguments `
            -Models " K2-Horizon-0.9B-GGUF, ,K2-Horizon-3.7B-GGUF " `
            -Lite $false
    ) `
    -Expected @(
        "--model",
        "K2-Horizon-0.9B-GGUF",
        "--model",
        "K2-Horizon-3.7B-GGUF"
    ) `
    -Case "explicit models"

Assert-Arguments `
    -Actual @(
        Get-LlamaCppValidationModelArguments `
            -Models "K2-Horizon-0.9B-GGUF,K2-Horizon-0.9B-GGUF" `
            -Lite $false
    ) `
    -Expected @(
        "--model",
        "K2-Horizon-0.9B-GGUF",
        "--model",
        "K2-Horizon-0.9B-GGUF"
    ) `
    -Case "duplicate models"

Assert-Arguments `
    -Actual @(Get-LlamaCppValidationModelArguments -Models " , " -Lite $true) `
    -Expected @("--lite") `
    -Case "lite mode"

$conflict = $null
try {
    Get-LlamaCppValidationModelArguments `
        -Models "K2-Horizon-0.9B-GGUF" `
        -Lite $true | Out-Null
} catch {
    $conflict = $_
}

if ($null -eq $conflict) {
    throw "Explicit models and lite mode must be rejected."
}

if ($conflict.Exception.Message -notmatch "mutually exclusive") {
    throw "Conflict error must explain that explicit models and lite mode are mutually exclusive."
}

$missingProfile = $null
try {
    Get-LlamaCppValidationModelArguments `
        -Models "K2-Horizon-0.9B-GGUF" `
        -CapabilityModels "K2-Horizon-0.9B-GGUF" | Out-Null
} catch {
    $missingProfile = $_
}

if ($null -eq $missingProfile) {
    throw "Capability models without a profile must be rejected."
}

$missingModel = $null
try {
    Get-LlamaCppValidationModelArguments `
        -Models "K2-Horizon-0.9B-GGUF" `
        -CapabilityProfile "k2-horizon-v1" | Out-Null
} catch {
    $missingModel = $_
}

if ($null -eq $missingModel) {
    throw "A capability profile without capability models must be rejected."
}

$unselectedModel = $null
try {
    Get-LlamaCppValidationModelArguments `
        -Models "K2-Horizon-0.9B-GGUF" `
        -CapabilityProfile "k2-horizon-v1" `
        -CapabilityModels "K2-Horizon-3.7B-GGUF" | Out-Null
} catch {
    $unselectedModel = $_
}

if ($null -eq $unselectedModel) {
    throw "Capability models outside the explicit selection must be rejected."
}
