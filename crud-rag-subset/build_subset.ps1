[CmdletBinding()]
param(
    [string]$InputPath,
    [int]$CorpusSize = 500
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$root = $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($InputPath)) {
    $InputPath = Join-Path $root 'split_merged.json'
    if (-not (Test-Path -LiteralPath $InputPath)) {
        $InputPath = Join-Path $root 'raw\split_merged.json'
    }
}

if (-not (Test-Path -LiteralPath $InputPath)) {
    throw "Input JSON was not found: $InputPath"
}

$utf8NoBom = New-Object System.Text.UTF8Encoding($false)

function Read-Utf8Json([string]$Path) {
    $text = [System.IO.File]::ReadAllText($Path, $utf8NoBom)
    return ($text | ConvertFrom-Json)
}

function Write-Utf8([string]$Path, [string]$Text) {
    [System.IO.File]::WriteAllText($Path, $Text, $utf8NoBom)
}

function Get-StableScore([string]$Value) {
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
        $bytes = [System.Text.Encoding]::UTF8.GetBytes($Value)
        $hash = $sha.ComputeHash($bytes)
        return [System.BitConverter]::ToUInt32($hash, 0)
    }
    finally {
        $sha.Dispose()
    }
}

if ($CorpusSize -lt 1) {
    throw 'CorpusSize must be positive.'
}

$data = Read-Utf8Json $InputPath
$eligible = @(
    @($data.questanswer_1doc) |
        Where-Object {
            -not [string]::IsNullOrWhiteSpace([string]$_.ID) -and
            -not [string]::IsNullOrWhiteSpace([string]$_.news1) -and
            -not [string]::IsNullOrWhiteSpace([string]$_.questions) -and
            -not [string]::IsNullOrWhiteSpace([string]$_.answers) -and
            ([string]$_.news1).Length -ge 500 -and
            ([string]$_.news1).Length -le 8000
        }
)

if ($eligible.Count -lt $CorpusSize) {
    throw "Only $($eligible.Count) eligible one-document QA records are available."
}

# Hash-based ordering is deterministic and spreads records across the source set.
$selected = @(
    $eligible |
        Sort-Object @{ Expression = { Get-StableScore ([string]$_.ID) }; Ascending = $true } |
        Select-Object -First $CorpusSize
)

$corpusDir = Join-Path $root 'corpus'
$evalDir = Join-Path $root 'eval'
New-Item -ItemType Directory -Force -Path $corpusDir, $evalDir | Out-Null

$corpusLines = @(
    foreach ($record in $selected) {
        $document = [ordered]@{
            id = [string]$record.ID
            title = ([string]$record.event).Trim()
            text = ([string]$record.news1).Trim()
            language = 'zh-CN'
            task = 'read'
            source = 'CRUD-RAG questanswer_1doc'
            source_record_id = [string]$record.ID
        }
        $document | ConvertTo-Json -Compress -Depth 5
    }
)
Write-Utf8 (Join-Path $corpusDir 'corpus.jsonl') (($corpusLines -join [Environment]::NewLine) + [Environment]::NewLine)

$evalLines = @(
    foreach ($record in $selected) {
        $item = [ordered]@{
            id = ('crud-rag-1doc-' + [string]$record.ID)
            question = ([string]$record.questions).Trim()
            answer = ([string]$record.answers).Trim()
            evidence_document_id = [string]$record.ID
            task = 'question_answering_1doc'
            language = 'zh-CN'
        }
        $item | ConvertTo-Json -Compress -Depth 5
    }
)
Write-Utf8 (Join-Path $evalDir 'qa_1doc.jsonl') (($evalLines -join [Environment]::NewLine) + [Environment]::NewLine)

$manifest = [ordered]@{
    dataset = 'CRUD-RAG'
    source_repository = 'https://github.com/IAAR-Shanghai/CRUD_RAG'
    source_file = 'data/crud_split/split_merged.json'
    source_url = 'https://raw.githubusercontent.com/IAAR-Shanghai/CRUD_RAG/main/data/crud_split/split_merged.json'
    language = 'zh-CN'
    task = 'question_answering_1doc'
    selection = [ordered]@{
        eligible_records = $eligible.Count
        selected_records = $selected.Count
        corpus_documents = $selected.Count
        ordering = 'ascending SHA-256(ID) score'
        filters = @('non-empty ID/news1/question/answer', 'news1 length 500..8000 characters')
    }
    files = @(
        'corpus/corpus.jsonl'
        'eval/qa_1doc.jsonl'
        'raw/split_merged.json'
    )
    generated_at_utc = (Get-Date).ToUniversalTime().ToString('o')
}
Write-Utf8 (Join-Path $root 'subset_manifest.json') ($manifest | ConvertTo-Json -Depth 8)

Write-Output "Selected $($selected.Count) documents and QA pairs."
