$ErrorActionPreference = "Stop"

function Convert-ToSource {
    param([string]$Text)
    $trimmed = $Text -replace "(\r?\n)+$",""
    if ([string]::IsNullOrWhiteSpace($trimmed)) {
        return @("")
    }
    return @([regex]::Split($trimmed, "\r?\n") | ForEach-Object { $_ + "`n" })
}

function Set-Cell {
    param(
        [object]$Notebook,
        [int]$Index,
        [string]$Type,
        [string]$Text
    )

    $Notebook.cells[$Index].cell_type = $Type
    $Notebook.cells[$Index].source = Convert-ToSource $Text

    if ($Type -eq "code") {
        if ($Notebook.cells[$Index].PSObject.Properties["execution_count"]) {
            $Notebook.cells[$Index].execution_count = $null
        } else {
            $Notebook.cells[$Index] | Add-Member -NotePropertyName execution_count -NotePropertyValue $null
        }

        if ($Notebook.cells[$Index].PSObject.Properties["outputs"]) {
            $Notebook.cells[$Index].outputs = @()
        } else {
            $Notebook.cells[$Index] | Add-Member -NotePropertyName outputs -NotePropertyValue @()
        }
    }
}
