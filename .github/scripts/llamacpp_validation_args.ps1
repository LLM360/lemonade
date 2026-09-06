function Get-LlamaCppValidationModelArguments {
    [CmdletBinding()]
    param(
        [AllowEmptyString()]
        [string] $Models = "",
        [bool] $Lite = $false,
        [AllowEmptyString()]
        [string] $CapabilityProfile = "",
        [AllowEmptyString()]
        [string] $CapabilityModels = ""
    )

    $modelIds = @(
        $Models -split "," |
            ForEach-Object { $_.Trim() } |
            Where-Object { $_ }
    )

    if ($modelIds.Count -gt 0 -and $Lite) {
        throw "Explicit models and lite mode are mutually exclusive."
    }

    $capabilityModelIds = @(
        $CapabilityModels -split "," |
            ForEach-Object { $_.Trim() } |
            Where-Object { $_ }
    )
    if (($CapabilityProfile -eq "") -ne ($capabilityModelIds.Count -eq 0)) {
        throw "Capability profile and capability models must be provided together."
    }
    if ($CapabilityProfile -and $CapabilityProfile -ne "k2-horizon-v1") {
        throw "Unsupported capability profile: $CapabilityProfile"
    }
    if ($modelIds.Count -gt 0) {
        foreach ($modelId in $capabilityModelIds) {
            if ($modelId -notin $modelIds) {
                throw "Capability model '$modelId' must be present in explicit models."
            }
        }
    }

    $validationArguments = @()
    foreach ($modelId in $modelIds) {
        $validationArguments += "--model"
        $validationArguments += $modelId
    }

    if ($Lite) {
        $validationArguments += "--lite"
    }

    if ($CapabilityProfile) {
        $validationArguments += "--capability-profile"
        $validationArguments += $CapabilityProfile
        foreach ($modelId in $capabilityModelIds) {
            $validationArguments += "--capability-model"
            $validationArguments += $modelId
        }
    }

    return $validationArguments
}
