$body = @{
    features = @(@{
        ammonia = 0.45
        bod = 4.5
        dissolved_oxygen = 4.0
        orthophosphate = 0.09
        ph = 5.8
        temperature = 28
        nitrogen = 9.0
        nitrate = 9.0
    })
} | ConvertTo-Json -Depth 5

Write-Host '=== Testing UNSAFE Water Sample (Bacterial Contamination) ===' -ForegroundColor Yellow
Write-Host 'URL: https://water-quality-api-symgat.onrender.com/predict' -ForegroundColor Cyan
Write-Host ''

try {
    $response = Invoke-RestMethod -Uri 'https://water-quality-api-symgat.onrender.com/predict' -Method Post -ContentType 'application/json' -Body $body
    
    Write-Host ' API Response Received' -ForegroundColor Green
    Write-Host ''
    Write-Host '--- Classification ---' -ForegroundColor Cyan
    Write-Host ('Classification: ' + $response.classification)
    Write-Host ('Forecast: ' + $response.forecast)
    Write-Host ('Risk Score: ' + $response.risk_score)
    Write-Host ''
    Write-Host '--- Predictions ---' -ForegroundColor Cyan
    Write-Host ('Water Quality Score: ' + $response.predictions[0])
    Write-Host ('Raw Score: ' + $response.raw_predictions[0])
    Write-Host ''
    Write-Host '--- SHAP Feature Importances (KEY TEST) ---' -ForegroundColor Yellow
    $response.explainability.feature_scores.PSObject.Properties | ForEach-Object {
        $value = [math]::Round($_.Value, 6)
        if ($value -ne 0) {
            Write-Host ('  ' + $_.Name + ': ' + $value) -ForegroundColor Green
        } else {
            Write-Host ('  ' + $_.Name + ': ' + $value) -ForegroundColor Red
        }
    }
    Write-Host ''
    Write-Host ('Main Feature: ' + $response.explainability.main_feature)
    Write-Host ('Main Feature Value: ' + $response.explainability.main_feature_value)
    Write-Host ''
    Write-Host '--- Explanation ---' -ForegroundColor Cyan
    Write-Host $response.explainability.plain_explanation
    Write-Host ''
    
    if ($response.explainability.feature_scores.ammonia -eq 0 -and $response.explainability.feature_scores.bod -eq 0) {
        Write-Host ' SHAP STILL RETURNING ZEROS!' -ForegroundColor Red
    } else {
        Write-Host ' SHAP WORKING! Non-zero feature importances detected!' -ForegroundColor Green
    }
} catch {
    Write-Host (' Error: ' + $_.Exception.Message) -ForegroundColor Red
}
