Add-Type -AssemblyName 'System.IO.Compression.FileSystem'
$zip = [System.IO.Compression.ZipFile]::OpenRead('c:\Code\Info.docx')
foreach ($entry in $zip.Entries) {
    if ($entry.Name -eq 'document.xml') {
        $stream = $entry.Open()
        $reader = New-Object System.IO.StreamReader($stream)
        $content = $reader.ReadToEnd()
        $reader.Close()
        $stream.Close()
        $cleaned = $content -replace '<[^>]+>', ' ' -replace '\s+', ' '
        Write-Output $cleaned
    }
}
$zip.Dispose()
