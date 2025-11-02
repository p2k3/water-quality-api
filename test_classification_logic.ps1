# Test script for unified classification logic
# Tests SAFE, MODERATE, and UNSAFE scenarios

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  TESTING CLASSIFICATION LOGIC" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

$baseUrl = "http://127.0.0.1:8000/predict"

# Test 1: SAFE Water (Compliant + Good Score)
Write-Host "TEST 1: SAFE WATER (Compliant)" -ForegroundColor Green
Write-Host "Expected: Classification=Compliant, Forecast=Safe, Card=Green`n" -ForegroundColor Gray

$safeBody = @{
    features = @(@{
        ammonia = 0.05
        bod = 1.2
        dissolved_oxygen = 8.5
        orthophosphate = 0.02
        ph = 7.2
        temperature = 18
        nitrogen = 1.5
        nitrate = 3.0
    })
} | ConvertTo-Json -Depth 5

try {
    $response = Invoke-RestMethod -Uri $baseUrl -Method Post -ContentType 'application/json' -Body $safeBody
    Write-Host "✓ Classification: $($response.classification)" -ForegroundColor $(if($response.classification -eq "Compliant"){"Green"}else{"Red"})
    Write-Host "✓ Forecast: $($response.forecast)" -ForegroundColor $(if($response.forecast -eq "Safe"){"Green"}else{"Red"})
    Write-Host "✓ Card Color: $($response.card_color)" -ForegroundColor $(if($response.card_color -eq "success"){"Green"}else{"Red"})
    Write-Host "✓ Confidence: $($response.confidence)"
    Write-Host "✓ Risk Score: $($response.risk_score.ToString('P1'))"
    Write-Host "✓ Parameter Violations: $($response.parameter_breakdown.Count)"
} catch {
    Write-Host "✗ ERROR: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host "`n----------------------------------------`n"

# Test 2: UNSAFE Water (Non-Compliant with Critical Violations)
Write-Host "TEST 2: UNSAFE WATER (Non-Compliant - Critical Violations)" -ForegroundColor Red
Write-Host "Expected: Classification=Non-Compliant, Forecast=Unsafe, Card=Red`n" -ForegroundColor Gray

$unsafeBody = @{
    features = @(@{
        ammonia = 0.45
        bod = 4.8
        dissolved_oxygen = 5.5
        orthophosphate = 0.095
        ph = 5.6
        temperature = 29
        nitrogen = 9.5
        nitrate = 45
    })
} | ConvertTo-Json -Depth 5

try {
    $response = Invoke-RestMethod -Uri $baseUrl -Method Post -ContentType 'application/json' -Body $unsafeBody
    Write-Host "✓ Classification: $($response.classification)" -ForegroundColor $(if($response.classification -eq "Non-Compliant"){"Green"}else{"Red"})
    Write-Host "✓ Forecast: $($response.forecast)" -ForegroundColor $(if($response.forecast -eq "Unsafe"){"Green"}else{"Red"})
    Write-Host "✓ Card Color: $($response.card_color)" -ForegroundColor $(if($response.card_color -eq "danger"){"Green"}else{"Red"})
    Write-Host "✓ Confidence: $($response.confidence)"
    Write-Host "✓ Risk Score: $($response.risk_score.ToString('P1'))"
    Write-Host "✓ Parameter Violations: $($response.parameter_breakdown.Count)" -ForegroundColor Yellow
    
    Write-Host "`nViolations Detail:" -ForegroundColor Yellow
    foreach ($violation in $response.parameter_breakdown) {
        $color = if($violation.severity -eq "CRITICAL"){"Red"}else{"Yellow"}
        Write-Host "  - $($violation.parameter): $($violation.value) [$($violation.severity)] ($($violation.deviation_percent.ToString('F1'))% deviation)" -ForegroundColor $color
    }
} catch {
    Write-Host "✗ ERROR: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host "`n----------------------------------------`n"

# Test 3: MODERATE Water (Compliant but Borderline)
Write-Host "TEST 3: MODERATE WATER (Compliant - Borderline)" -ForegroundColor Yellow
Write-Host "Expected: Classification=Compliant, Forecast=Moderate, Card=Orange`n" -ForegroundColor Gray

$moderateBody = @{
    features = @(@{
        ammonia = 0.25
        bod = 3.0
        dissolved_oxygen = 6.2
        orthophosphate = 0.06
        ph = 6.8
        temperature = 24
        nitrogen = 5.5
        nitrate = 20
    })
} | ConvertTo-Json -Depth 5

try {
    $response = Invoke-RestMethod -Uri $baseUrl -Method Post -ContentType 'application/json' -Body $moderateBody
    Write-Host "✓ Classification: $($response.classification)" -ForegroundColor $(if($response.classification -eq "Compliant"){"Green"}else{"Red"})
    Write-Host "✓ Forecast: $($response.forecast)" -ForegroundColor $(if($response.forecast -eq "Moderate"){"Green"}else{"Red"})
    Write-Host "✓ Card Color: $($response.card_color)" -ForegroundColor $(if($response.card_color -eq "warning"){"Green"}else{"Red"})
    Write-Host "✓ Confidence: $($response.confidence)"
    Write-Host "✓ Risk Score: $($response.risk_score.ToString('P1'))"
    Write-Host "✓ Parameter Violations: $($response.parameter_breakdown.Count)"
    Write-Host "`n✓ Explanation: $($response.explainability.plain_explanation)" -ForegroundColor Cyan
} catch {
    Write-Host "✗ ERROR: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  TEST COMPLETE" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan
