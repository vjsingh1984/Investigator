# InvestiGator Images Directory

This directory contains visual assets for enhanced PDF report generation.

## Directory Structure

```
images/
├── icons/           # Recommendation icons (buy, sell, hold, etc.)
├── charts/          # Chart templates and backgrounds
└── backgrounds/     # Report page backgrounds and watermarks
```

## Icon Color Scheme

- **STRONG BUY**: Deep Green (#006400)
- **BUY**: Green (#228B22)
- **HOLD**: Gold/Yellow (#FFD700)
- **SELL**: Orange (#FF8C00)
- **STRONG SELL**: Red (#DC143C)

## Required Icons (to be added)

1. **Recommendation Icons** (64x64 PNG):
   - strong_buy.png
   - buy.png
   - hold.png
   - sell.png
   - strong_sell.png

2. **Confidence Level Icons** (32x32 PNG):
   - high_confidence.png
   - medium_confidence.png
   - low_confidence.png

3. **Analysis Type Icons** (48x48 PNG):
   - fundamental_analysis.png
   - technical_analysis.png
   - synthesis.png

4. **Health Indicators** (48x48 PNG):
   - excellent_health.png (green checkmark)
   - good_health.png (blue thumbs up)
   - fair_health.png (yellow warning)
   - poor_health.png (red alert)

5. **Trend Arrows** (32x32 PNG):
   - strong_uptrend.png
   - uptrend.png
   - sideways.png
   - downtrend.png
   - strong_downtrend.png

## Chart Templates

The charts directory should contain:
- Bollinger bands chart template
- RSI indicator template
- Volume/OBV template
- Fundamental health radar chart template

## Usage in PDF

These images are used by the ReportGenerator class to create visually appealing PDF reports with:
- Color-coded recommendations
- Visual health indicators
- Technical chart overlays
- Professional watermarks

## License

All images in this directory are subject to the Apache License 2.0.
Copyright (c) 2025 Vijaykumar Singh