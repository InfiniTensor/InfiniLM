<#
.SYNOPSIS
  Archive the MiniMax work into your own GitHub branches and tags (InfiniCore + InfiniLM).

.DESCRIPTION
  - InfiniCore: adds remote `myfork` (https://github.com/<GitHubUser>/InfiniCore.git),
    creates the archive branch on base 35b46277, commits, tags and pushes it.
  - InfiniLM: creates the archive branch on origin (your own fork), commits, tags and pushes it.
  - Only the explicitly listed MiniMax files are staged; the staging area is verified
    before committing and the script aborts on any unexpected file.

.PARAMETER DryRun
  Prints the commands instead of running the mutating ones; `git add -n` validates paths for real.

.PARAMETER SkipPush
  Commits and tags locally, but does not push.

.EXAMPLE
  pwsh -File docs/minimax/archive-to-fork.ps1 -DryRun
  pwsh -File docs/minimax/archive-to-fork.ps1
#>
[CmdletBinding()]
param(
    [string]$GitHubUser = 'y258dd',
    [string]$InfiniCorePath = 'C:\Users\yht13\InfiniCore',
    [string]$InfiniLMPath = 'C:\Users\yht13\InfiniLM',
    [switch]$DryRun,
    [switch]$SkipPush
)

$ErrorActionPreference = 'Stop'

$InfiniCoreExpectedHead = '35b46277bd666772c11bb417ad4231c5be492822'
$InfiniCoreBranch = 'archive/minimax-lightning-attn'
$InfiniCoreTag = 'minimax-lightning-attn-35b46277'
$InfiniCoreSubject = 'feat(infiniop): add lightning_attention op (CPU + NVIDIA) for MiniMax'
$InfiniCoreBody = "Implements indexed-pool lightning attention (MiniMax-01 style):`n  S <- ratio * S + k^T v ; o = q * S   with ratio[h] = exp(-slope[h])`nCPU reference implementation plus a CUDA kernel, wired through the`ninfiniop C API, the infinicore C++ op layer and the Python bindings."
$InfiniCorePaths = @(
    '.gitignore',
    'include/infiniop.h',
    'include/infinicore/ops.hpp',
    'src/infinicore/pybind11/ops.hpp',
    'python/infinicore/__init__.py',
    'include/infiniop/ops/lightning_attention.h',
    'include/infinicore/ops/lightning_attention.hpp',
    'src/infiniop/ops/lightning_attention',
    'src/infinicore/ops/lightning_attention',
    'src/infinicore/pybind11/ops/lightning_attention.hpp',
    'python/infinicore/ops/lightning_attention.py'
)

$InfiniLMBranch = 'archive/minimax'
$InfiniLMTag = 'minimax-support'
$InfiniLMSubject = 'feat(minimax): support MiniMax-Text-01 (lightning attention + MoE)'
$InfiniLMBody = "- csrc/models/minimax: MiniMax model (hybrid lightning/softmax attention,`n  block-sparse MoE, dense fallback), registered as minimax/minimax_m2`n- python/infinilm/modeling_utils.py: _remap_minimax weight remapper`n- test/models/minimax: op unit test, model smoke test, HF-aligned E2E test`n- docs/minimax: porting guide, handover notes and the InfiniCore op patch"
$InfiniLMPaths = @(
    '.gitignore',
    'python/infinilm/modeling_utils.py',
    'csrc/models/minimax',
    'test/models/minimax',
    'docs/minimax'
)

function Write-Step {
    param([string]$Text)
    Write-Host ''
    Write-Host "=== $Text ===" -ForegroundColor Cyan
}

function Invoke-Git {
    param(
        [Parameter(Mandatory)][string]$Repo,
        [Parameter(Mandatory)][string[]]$GitArgs,
        [switch]$Mutating
    )
    $display = 'git -C "' + $Repo + '" ' + ($GitArgs -join ' ')
    if ($DryRun -and $Mutating) {
        Write-Host "  [dry-run] $display" -ForegroundColor Yellow
        return @()
    }
    Write-Host "  > $display" -ForegroundColor DarkGray
    $output = & git -C $Repo @GitArgs
    if ($LASTEXITCODE -ne 0) { throw "git command failed: $display" }
    return $output
}

function Assert-Repo {
    param([string]$Path, [string]$Name)
    if (-not (Test-Path (Join-Path $Path '.git'))) { throw "$Name is not a git repository: $Path" }
}

function Assert-StagedSet {
    param([string]$Repo, [string[]]$Expected)
    $staged = @(& git -C $Repo diff --cached --name-only | Where-Object { $_ })
    $unexpected = @($staged | Where-Object {
        $file = $_
        -not ($Expected | Where-Object { $file -eq $_ -or $file.StartsWith($_ + '/') })
    })
    if ($unexpected.Count -gt 0) {
        throw ("Unexpected staged files, aborting:`n  " + ($unexpected -join "`n  "))
    }
    Write-Host ("  staged files: {0} (all within the expected list)" -f $staged.Count) -ForegroundColor Green
}

function Ensure-Branch {
    param([string]$Repo, [string]$Branch)
    $existing = @(& git -C $Repo branch --list $Branch | Where-Object { $_ })
    if ($existing.Count -gt 0) {
        Write-Host "  branch already exists, checking it out: $Branch" -ForegroundColor Yellow
        Invoke-Git -Repo $Repo -GitArgs @('checkout', $Branch) -Mutating | Out-Null
    } else {
        Invoke-Git -Repo $Repo -GitArgs @('checkout', '-b', $Branch) -Mutating | Out-Null
    }
}

function Commit-If-Needed {
    param(
        [string]$Repo,
        [string]$Subject,
        [string]$Body,
        [string[]]$Paths
    )
    if ($DryRun) {
        Invoke-Git -Repo $Repo -GitArgs (@('add', '-n', '--') + $Paths) | Out-Null
    } else {
        Invoke-Git -Repo $Repo -GitArgs (@('add', '--') + $Paths) -Mutating | Out-Null
        Assert-StagedSet -Repo $Repo -Expected $Paths
        $stagedCount = @(& git -C $Repo diff --cached --name-only | Where-Object { $_ }).Count
        if ($stagedCount -eq 0) {
            Write-Host '  nothing to commit (already committed?), skipping commit.' -ForegroundColor Yellow
            return
        }
    }
    Invoke-Git -Repo $Repo -GitArgs @('commit', '-m', $Subject, '-m', $Body) -Mutating | Out-Null
}

function Tag-If-Missing {
    param([string]$Repo, [string]$Tag)
    $existing = @(& git -C $Repo tag --list $Tag | Where-Object { $_ })
    if ($existing.Count -gt 0) {
        Write-Host "  tag already exists, skipping: $Tag" -ForegroundColor Yellow
        return
    }
    Invoke-Git -Repo $Repo -GitArgs @('tag', $Tag) -Mutating | Out-Null
}

# ---------------------------------------------------------------- InfiniCore
Write-Step "InfiniCore: $InfiniCorePath"
Assert-Repo -Path $InfiniCorePath -Name 'InfiniCore'

$head = (& git -C $InfiniCorePath rev-parse HEAD).Trim()
if ($head -ne $InfiniCoreExpectedHead) {
    Write-Host "  WARNING: HEAD = $head, expected base $InfiniCoreExpectedHead" -ForegroundColor Yellow
    Write-Host "           (ignore if you already archived; otherwise check for stray commits)" -ForegroundColor Yellow
} else {
    Write-Host "  base commit confirmed: $head" -ForegroundColor Green
}

$forkUrl = "https://github.com/$GitHubUser/InfiniCore.git"
$remotes = @(& git -C $InfiniCorePath remote | Where-Object { $_ })
if ($remotes -notcontains 'myfork') {
    Invoke-Git -Repo $InfiniCorePath -GitArgs @('remote', 'add', 'myfork', $forkUrl) -Mutating | Out-Null
} else {
    $existingUrl = (& git -C $InfiniCorePath remote get-url myfork).Trim()
    if ($existingUrl -ne $forkUrl) {
        Write-Host "  NOTE: myfork exists with url $existingUrl (expected $forkUrl)" -ForegroundColor Yellow
    }
    Write-Host "  myfork configured: $existingUrl" -ForegroundColor Green
}

Ensure-Branch -Repo $InfiniCorePath -Branch $InfiniCoreBranch
Commit-If-Needed -Repo $InfiniCorePath -Subject $InfiniCoreSubject -Body $InfiniCoreBody -Paths $InfiniCorePaths
Tag-If-Missing -Repo $InfiniCorePath -Tag $InfiniCoreTag

if (-not $SkipPush) {
    Invoke-Git -Repo $InfiniCorePath -GitArgs @('push', '-u', 'myfork', $InfiniCoreBranch) -Mutating | Out-Null
    Invoke-Git -Repo $InfiniCorePath -GitArgs @('push', 'myfork', $InfiniCoreTag) -Mutating | Out-Null
}

# ----------------------------------------------------------------- InfiniLM
Write-Step "InfiniLM: $InfiniLMPath"
Assert-Repo -Path $InfiniLMPath -Name 'InfiniLM'

$originUrl = (& git -C $InfiniLMPath remote get-url origin).Trim()
Write-Host "  origin = $originUrl"
if ($originUrl -notmatch [regex]::Escape($GitHubUser)) {
    Write-Host "  WARNING: origin does not contain account $GitHubUser; please confirm this is your own fork." -ForegroundColor Yellow
} else {
    Write-Host "  confirmed as your own fork." -ForegroundColor Green
}

Ensure-Branch -Repo $InfiniLMPath -Branch $InfiniLMBranch
Commit-If-Needed -Repo $InfiniLMPath -Subject $InfiniLMSubject -Body $InfiniLMBody -Paths $InfiniLMPaths
Tag-If-Missing -Repo $InfiniLMPath -Tag $InfiniLMTag

if (-not $SkipPush) {
    Invoke-Git -Repo $InfiniLMPath -GitArgs @('push', '-u', 'origin', $InfiniLMBranch) -Mutating | Out-Null
    Invoke-Git -Repo $InfiniLMPath -GitArgs @('push', 'origin', $InfiniLMTag) -Mutating | Out-Null
}

# ------------------------------------------------------------------- Summary
Write-Step 'Done'
if ($DryRun) {
    Write-Host 'This was a dry run; nothing was changed except read-only checks. Re-run without -DryRun to execute.' -ForegroundColor Yellow
} else {
    Write-Host "  InfiniCore: branch $InfiniCoreBranch , tag $InfiniCoreTag (remote: myfork)" -ForegroundColor Green
    Write-Host "  InfiniLM  : branch $InfiniLMBranch , tag $InfiniLMTag (remote: origin)" -ForegroundColor Green
    if ($SkipPush) {
        Write-Host '  Push skipped (-SkipPush). Push manually with:' -ForegroundColor Yellow
        Write-Host "    git -C `"$InfiniCorePath`" push -u myfork $InfiniCoreBranch"
        Write-Host "    git -C `"$InfiniCorePath`" push myfork $InfiniCoreTag"
        Write-Host "    git -C `"$InfiniLMPath`" push -u origin $InfiniLMBranch"
        Write-Host "    git -C `"$InfiniLMPath`" push origin $InfiniLMTag"
    }
    Write-Host ''
    Write-Host '  Next (server compile check): InfiniCore base 35b46277 + docs/minimax/lightning-attention-infinicore.patch' -ForegroundColor Cyan
}