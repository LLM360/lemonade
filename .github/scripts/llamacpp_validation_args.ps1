function Get-LlamaCppValidationModelArguments {
    [CmdletBinding()]
    param(
        [AllowEmptyString()]
        [string] $Models = "",
        [bool] $Lite = $false
    )

    $modelIds = @(
        $Models -split "," |
            ForEach-Object { $_.Trim() } |
            Where-Object { $_ }
    )

    if ($modelIds.Count -gt 0 -and $Lite) {
        throw "Explicit models and lite mode are mutually exclusive."
    }

    $validationArguments = @()
    foreach ($modelId in $modelIds) {
        $validationArguments += "--model"
        $validationArguments += $modelId
    }

    if ($Lite) {
        $validationArguments += "--lite"
    }

    return $validationArguments
}
