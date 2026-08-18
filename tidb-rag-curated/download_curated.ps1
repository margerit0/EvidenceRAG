[CmdletBinding()]
param(
    [int]$MaxAttempts = 4
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$root = $PSScriptRoot
$manifestPath = Join-Path $root 'selected_manifest.json'
$documentsRoot = Join-Path $root 'documents'
$corpusManifestPath = Join-Path $root 'corpus_manifest.jsonl'
$reportPath = Join-Path $root 'download_report.json'

if (-not (Test-Path -LiteralPath $manifestPath)) {
    throw "Selection manifest was not found: $manifestPath"
}
if ($MaxAttempts -lt 1) {
    throw 'MaxAttempts must be at least 1.'
}

function Get-Hex([byte[]]$Bytes) {
    return ([System.BitConverter]::ToString($Bytes)).Replace('-', '').ToLowerInvariant()
}

function Get-GitBlobSha([byte[]]$Content) {
    $header = [System.Text.Encoding]::ASCII.GetBytes("blob $($Content.Length)`0")
    $payload = New-Object byte[] ($header.Length + $Content.Length)
    [System.Buffer]::BlockCopy($header, 0, $payload, 0, $header.Length)
    [System.Buffer]::BlockCopy($Content, 0, $payload, $header.Length, $Content.Length)
    $sha1 = [System.Security.Cryptography.SHA1]::Create()
    try {
        return Get-Hex ($sha1.ComputeHash($payload))
    }
    finally {
        $sha1.Dispose()
    }
}

function Get-Sha256([byte[]]$Content) {
    $sha256 = [System.Security.Cryptography.SHA256]::Create()
    try {
        return Get-Hex ($sha256.ComputeHash($Content))
    }
    finally {
        $sha256.Dispose()
    }
}

function Get-RemoteBytes {
    param(
        [System.Net.Http.HttpClient]$Client,
        [string]$Url,
        [int]$Attempts
    )
    for ($attempt = 1; $attempt -le $Attempts; $attempt++) {
        try {
            return $Client.GetByteArrayAsync($Url).GetAwaiter().GetResult()
        }
        catch {
            if ($attempt -eq $Attempts) { throw }
            Start-Sleep -Seconds ([math]::Min(5, $attempt))
        }
    }
}

Add-Type -AssemblyName System.Net.Http
[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.SecurityProtocolType]::Tls12
$manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
$documents = @($manifest.documents)
[void][System.IO.Directory]::CreateDirectory($documentsRoot)

$client = [System.Net.Http.HttpClient]::new()
$client.Timeout = [TimeSpan]::FromSeconds(60)
$client.DefaultRequestHeaders.UserAgent.ParseAdd('tidb-rag-curator/1.0')
$rows = New-Object 'System.Collections.Generic.List[object]'
$downloaded = 0
$reused = 0
$totalBytes = [int64]0

try {
    for ($index = 0; $index -lt $documents.Count; $index++) {
        $doc = $documents[$index]
        $relativePath = "documents/$($doc.collection)/$($doc.path)"
        $localPath = Join-Path $root ($relativePath.Replace('/', [System.IO.Path]::DirectorySeparatorChar))
        $parent = Split-Path -Parent $localPath
        [void][System.IO.Directory]::CreateDirectory($parent)

        $bytes = $null
        if (Test-Path -LiteralPath $localPath -PathType Leaf) {
            $existing = [System.IO.File]::ReadAllBytes($localPath)
            if ($existing.Length -eq [int64]$doc.size -and (Get-GitBlobSha $existing) -eq [string]$doc.sha) {
                $bytes = $existing
                $reused++
            }
        }
        if ($null -eq $bytes) {
            $bytes = Get-RemoteBytes -Client $client -Url ([string]$doc.raw_url) -Attempts $MaxAttempts
            if ($bytes.Length -ne [int64]$doc.size) {
                throw "Size mismatch for $($doc.path): expected $($doc.size), got $($bytes.Length)."
            }
            $blobSha = Get-GitBlobSha $bytes
            if ($blobSha -ne [string]$doc.sha) {
                throw "Git blob SHA mismatch for $($doc.path): expected $($doc.sha), got $blobSha."
            }
            [System.IO.File]::WriteAllBytes($localPath, $bytes)
            $downloaded++
        }

        $sha256 = Get-Sha256 $bytes
        $totalBytes += $bytes.Length
        $rows.Add([pscustomobject][ordered]@{
            id = [string]$doc.sha
            path = [string]$doc.path
            local_path = $relativePath
            collection = [string]$doc.collection
            theme = [string]$doc.theme
            category = [string]$doc.category
            language = 'zh-CN'
            bytes = [int64]$bytes.Length
            git_blob_sha1 = [string]$doc.sha
            sha256 = $sha256
            source_url = [string]$doc.source_url
            source_commit = [string]$manifest.commit
        })

        if ((($index + 1) % 50) -eq 0 -or ($index + 1) -eq $documents.Count) {
            Write-Output "Verified $($index + 1)/$($documents.Count) documents."
        }
    }
}
finally {
    $client.Dispose()
}

$utf8 = New-Object System.Text.UTF8Encoding($false)
$jsonLines = @($rows.ToArray() | ForEach-Object { $_ | ConvertTo-Json -Compress -Depth 5 })
[System.IO.File]::WriteAllLines($corpusManifestPath, $jsonLines, $utf8)

$collectionSummary = @($rows.ToArray() | Group-Object collection | Sort-Object Name | ForEach-Object {
    [ordered]@{
        collection = $_.Name
        documents = $_.Count
        bytes = [int64](($_.Group | Measure-Object bytes -Sum).Sum)
    }
})
$themeSummary = @($rows.ToArray() | Group-Object theme | Sort-Object Name | ForEach-Object {
    [ordered]@{
        theme = $_.Name
        documents = $_.Count
    }
})
$report = [ordered]@{
    dataset = [string]$manifest.dataset
    source_repository = [string]$manifest.source_repository
    source_commit = [string]$manifest.commit
    total_documents = $rows.Count
    downloaded_this_run = $downloaded
    reused_this_run = $reused
    total_bytes = $totalBytes
    integrity = 'Every file matched the Git tree blob SHA-1 and expected byte size; SHA-256 recorded in corpus_manifest.jsonl.'
    collections = $collectionSummary
    themes = $themeSummary
    generated_at_utc = (Get-Date).ToUniversalTime().ToString('o')
}
[System.IO.File]::WriteAllText($reportPath, ($report | ConvertTo-Json -Depth 7), $utf8)

Write-Output "Completed: $($rows.Count) verified documents, $downloaded downloaded, $reused reused."
