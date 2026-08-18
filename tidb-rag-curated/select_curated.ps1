[CmdletBinding()]
param(
    [int]$Total = 500,
    [int]$ReleaseCount = 50
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$root = $PSScriptRoot
$treePath = Join-Path $root 'tree.json'
$manifestPath = Join-Path $root 'selected_manifest.json'
$pathsPath = Join-Path $root 'selected_paths.txt'

if (-not (Test-Path -LiteralPath $treePath)) {
    throw "Tree metadata was not found: $treePath"
}
if ($Total -lt 1 -or $ReleaseCount -lt 0 -or $ReleaseCount -ge $Total) {
    throw 'Total must be positive and ReleaseCount must be between 0 and Total - 1.'
}

$tree = Get-Content -LiteralPath $treePath -Raw -Encoding UTF8 | ConvertFrom-Json
$commit = [string]$tree.sha

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

function Get-Category([string]$Path) {
    if ($Path.Contains('/')) {
        return $Path.Split('/')[0]
    }
    return 'root'
}

function Get-PathScore([string]$Path) {
    $score = 0
    $lower = $Path.ToLowerInvariant()
    $weights = @(
        @{ Pattern = 'overview|quick-start|quickstart|getting-started|guide|tutorial|example'; Weight = 35 }
        @{ Pattern = 'deploy|deployment|topology|install|configuration|config|setup|initialize'; Weight = 30 }
        @{ Pattern = 'troubleshoot|diagnos|error|failure|faq|monitor|alert|slow|oom|lock|hot-spot'; Weight = 30 }
        @{ Pattern = 'vector|embedding|hybrid-search|full-text|ai/|python|java|go|jdbc|sdk|integration'; Weight = 28 }
        @{ Pattern = 'backup|restore|br/|dumpling|lightning|import|export|replicat|changefeed|cdc'; Weight = 25 }
        @{ Pattern = 'select|insert|update|delete|create|alter|transaction|explain|index|partition|statement'; Weight = 20 }
        @{ Pattern = 'best-practice|security|performance|tuning|capacity|production'; Weight = 18 }
        @{ Pattern = 'command-line|flags|reference|syntax|variables|functions|operators'; Weight = 8 }
        @{ Pattern = 'glossary|terminology|changelog|release'; Weight = -8 }
    )
    foreach ($item in $weights) {
        if ($lower -match $item.Pattern) {
            $score += [int]$item.Weight
        }
    }
    return $score
}

function Is-UsableMarkdown($Entry) {
    $path = [string]$Entry.path
    if ($Entry.type -ne 'blob' -or $path -notlike '*.md') { return $false }
    if ($path -match '^(\.agents|\.github|\.vaunt|scripts|media|templates)/') { return $false }
    if ($path -match '(^|/)(CONTRIBUTING|README|TOC[^/]*|_docHome|_index)\.md$') { return $false }
    return $true
}

$docs = @($tree.tree | Where-Object { Is-UsableMarkdown $_ } | ForEach-Object {
    [pscustomobject]@{
        path = [string]$_.path
        sha = [string]$_.sha
        size = [int64]$_.size
        category = Get-Category ([string]$_.path)
        stable = Get-StableScore ([string]$_.path)
    }
})

$selected = New-Object 'System.Collections.Generic.List[object]'
$seen = New-Object 'System.Collections.Generic.HashSet[string]'
$planResults = New-Object 'System.Collections.Generic.List[object]'

function Add-Plan {
    param(
        [string]$Collection,
        [string]$Theme,
        [int]$Target,
        [object[]]$Candidates
    )
    $available = @($Candidates | Where-Object { -not $seen.Contains([string]$_.path) })
    $ranked = @($available | Sort-Object `
        @{ Expression = { Get-PathScore ([string]$_.path) }; Descending = $true }, `
        @{ Expression = { $_.stable }; Ascending = $true }, `
        @{ Expression = { $_.path }; Ascending = $true })
    $take = @($ranked | Select-Object -First $Target)
    foreach ($doc in $take) {
        if ($seen.Add([string]$doc.path)) {
            $selected.Add([pscustomobject]@{
                path = $doc.path
                sha = $doc.sha
                size = $doc.size
                category = $doc.category
                collection = $Collection
                theme = $Theme
                source_url = "https://github.com/pingcap/docs-cn/blob/$commit/$($doc.path)"
                raw_url = "https://raw.githubusercontent.com/pingcap/docs-cn/$commit/$($doc.path)"
            })
        }
    }
    $planResults.Add([pscustomobject]@{
        collection = $Collection
        theme = $Theme
        target = $Target
        available = $available.Count
        selected = $take.Count
    })
}

$all = @($docs)
$ai = @($all | Where-Object { $_.category -eq 'ai' -or ($_.category -eq 'root' -and $_.path -match '(?i)(vector|embedding|hybrid-search|full-text-search|ai)') })
$deploy = @($all | Where-Object {
    $_.category -in @('tiup','config-templates') -or
    ($_.category -eq 'root' -and $_.path -match '(?i)(deploy|deployment|topology|configuration|config|hardware|kubernetes|operator|tls|certificate|security)')
})
$troubleshoot = @($all | Where-Object {
    $_.category -in @('clinic','dashboard') -or
    ($_.category -eq 'root' -and $_.path -match '(?i)(troubleshoot|diagnos|alert|monitor|slow|oom|lock|hot-spot|daily-check|error)')
})
$backup = @($all | Where-Object {
    $_.category -in @('br','tidb-lightning') -or
    ($_.category -eq 'root' -and $_.path -match '(?i)(backup|restore|dumpling|lightning|import|export)')
})
$develop = @($all | Where-Object {
    $_.category -eq 'develop' -or
    ($_.category -eq 'root' -and $_.path -match '(?i)(quick-start|integration|application|driver|sdk|develop|connect)')
})
$ecosystem = @($all | Where-Object {
    $_.category -in @('dm','ticdc','tiflash','tiproxy','storage-engine','sync-diff-inspector')
})
$sql = @($all | Where-Object {
    $_.category -in @('sql-statements','functions-and-operators','information-schema','mysql-schema','performance-schema','sys-schema') -or
    ($_.category -eq 'root' -and $_.path -match '(?i)(sql|statement|transaction|explain|index|partition|data-type|character-set|collation|optimizer|system-variable|table|view|constraint)')
})
$best = @($all | Where-Object { $_.category -in @('best-practices','faq','benchmark') })

# Evergreen targets total 450. Plans are applied in this order so high-value
# topics claim overlapping root-level documents first.
Add-Plan -Collection 'dev_reference' -Theme 'ai_vector_functions' -Target 35 -Candidates $ai
Add-Plan -Collection 'core_ops' -Theme 'deploy_config_tiup' -Target 75 -Candidates $deploy
Add-Plan -Collection 'core_ops' -Theme 'troubleshoot_monitoring' -Target 55 -Candidates $troubleshoot
Add-Plan -Collection 'core_ops' -Theme 'backup_import' -Target 35 -Candidates $backup
Add-Plan -Collection 'core_ops' -Theme 'ecosystem_components' -Target 45 -Candidates $ecosystem
Add-Plan -Collection 'dev_reference' -Theme 'develop_app' -Target 55 -Candidates $develop
Add-Plan -Collection 'dev_reference' -Theme 'sql_reference' -Target 110 -Candidates $sql
Add-Plan -Collection 'dev_reference' -Theme 'best_practices_faq_benchmark' -Target 25 -Candidates $best

$evergreenTarget = $Total - $ReleaseCount
$evergreenSelected = @($selected | Where-Object { $_.collection -ne 'temporal_releases' }).Count
$remainingEvergreen = $evergreenTarget - $evergreenSelected
if ($remainingEvergreen -gt 0) {
    $fallback = @($all | Where-Object { $_.category -ne 'releases' -and -not $seen.Contains([string]$_.path) })
    Add-Plan -Collection 'dev_reference' -Theme 'fallback_evergreen' -Target $remainingEvergreen -Candidates $fallback
}

$releaseDocs = @($all | Where-Object { $_.category -eq 'releases' })
$releaseBase = @($releaseDocs | Where-Object { $_.path -match '(?i)releases/(?:_index|versioning|release-timeline)\.md$' })
$releaseVersioned = @($releaseDocs | Where-Object { $_.path -match '(?i)releases/release-\d+\.\d+' })
# Prefer one representative (latest patch) for each minor version before
# spending the remaining slots on additional recent patch releases. This
# prevents the temporal collection from being dominated by one old patch
# series (for example, every 6.5.x release).
function Get-ReleaseVersionScore([string]$Path) {
    $m = [regex]::Match($Path, 'release-(\d+)\.(\d+)(?:\.(\d+))?')
    if ($m.Success) {
        $patch = if ($m.Groups[3].Success) { [int]$m.Groups[3].Value } else { 0 }
        return ([int]$m.Groups[1].Value * 1000000 + [int]$m.Groups[2].Value * 1000 + $patch)
    }
    return 0
}
$releaseRanked = @($releaseVersioned | Sort-Object `
    @{ Expression = { Get-ReleaseVersionScore ([string]$_.path) }; Descending = $true },
    @{ Expression = { $_.stable }; Ascending = $true })
$releaseNeed = $ReleaseCount
$releaseTake = @($releaseBase | Select-Object -First ([math]::Min($releaseNeed, $releaseBase.Count)))
$releaseNeed = $releaseNeed - $releaseTake.Count
# Select the newest document from each major.minor family first.
$minorRepresentatives = @($releaseRanked | Group-Object {
    $m = [regex]::Match([string]$_.path, 'release-(\d+)\.(\d+)')
    if ($m.Success) { "$($m.Groups[1].Value).$($m.Groups[2].Value)" } else { [string]$_.path }
} | ForEach-Object {
    @($_.Group | Sort-Object `
        @{ Expression = { Get-ReleaseVersionScore ([string]$_.path) }; Descending = $true },
        @{ Expression = { $_.stable }; Ascending = $true } | Select-Object -First 1)
})
$releaseTake += @($minorRepresentatives | Sort-Object `
    @{ Expression = { Get-ReleaseVersionScore ([string]$_.path) }; Descending = $true },
    @{ Expression = { $_.stable }; Ascending = $true } | Select-Object -First $releaseNeed)
$releaseNeed = $ReleaseCount - $releaseTake.Count
if ($releaseNeed -gt 0) {
    $releaseTake += @($releaseRanked | Where-Object { $_.path -notin @($releaseTake.path) } | Select-Object -First $releaseNeed)
}
if ($releaseTake.Count -lt $ReleaseCount) {
    $releaseFallback = @($releaseDocs | Where-Object { $_.path -notin @($releaseTake.path) } | Sort-Object stable | Select-Object -First ($ReleaseCount - $releaseTake.Count))
    $releaseTake += $releaseFallback
}
foreach ($doc in $releaseTake) {
    if ($seen.Add([string]$doc.path)) {
        $selected.Add([pscustomobject]@{
            path = $doc.path
            sha = $doc.sha
            size = $doc.size
            category = $doc.category
            collection = 'temporal_releases'
            theme = 'version_history'
            source_url = "https://github.com/pingcap/docs-cn/blob/$commit/$($doc.path)"
            raw_url = "https://raw.githubusercontent.com/pingcap/docs-cn/$commit/$($doc.path)"
        })
    }
}

if ($selected.Count -lt $Total) {
    throw "Only $($selected.Count) documents were selected; expected $Total."
}
if ($selected.Count -gt $Total) {
    $selected = [System.Collections.Generic.List[object]]( @($selected | Select-Object -First $Total) )
}

$selectedArray = [object[]]$selected.ToArray()
$planArray = [object[]]$planResults.ToArray()
$summary = @($selectedArray | Group-Object collection,theme | ForEach-Object {
    [pscustomobject]@{ Group = $_.Name; Count = $_.Count }
})
$evergreenRows = @($selectedArray | Where-Object { $_.collection -ne 'temporal_releases' })
$releaseRows = @($selectedArray | Where-Object { $_.collection -eq 'temporal_releases' })
$manifest = [ordered]@{
    dataset = 'TiDB Chinese RAG curated corpus'
    source_repository = 'https://github.com/pingcap/docs-cn'
    commit = $commit
    commit_url = "https://github.com/pingcap/docs-cn/tree/$commit"
    language = 'zh-CN'
    selection_policy = 'stratified topic quotas; stable path hash tie-break; release documents isolated'
    total_selected = $selected.Count
    evergreen_selected = $evergreenRows.Count
    release_selected = $releaseRows.Count
    plan_results = $planArray
    summary = $summary
    documents = @($selectedArray | Sort-Object collection,theme,path)
    generated_at_utc = (Get-Date).ToUniversalTime().ToString('o')
}
$json = $manifest | ConvertTo-Json -Depth 8
[System.IO.File]::WriteAllText($manifestPath, $json, (New-Object System.Text.UTF8Encoding($false)))
[System.IO.File]::WriteAllLines($pathsPath, @($selectedArray | Sort-Object collection,theme,path | ForEach-Object { $_.path }), (New-Object System.Text.UTF8Encoding($false)))

Write-Output "Selected $($selected.Count) documents ($($evergreenRows.Count) evergreen + $($releaseRows.Count) releases)."
$planResults | Format-Table | Out-String | Write-Output
