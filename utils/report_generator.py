#!/usr/bin/env python3
"""
InvestiGator - PDF Report Generation Module
Copyright (c) 2025 Vijaykumar Singh
Licensed under the Apache License 2.0

PDF Report Generation Module for InvestiGator
Handles creation of investment analysis reports
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass
import json
import re
import markdown

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, 
    PageBreak, Image, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_JUSTIFY, TA_LEFT
from reportlab.pdfgen import canvas
from reportlab.platypus.flowables import Flowable
from reportlab.graphics.shapes import Drawing, Circle, Rect
from reportlab.graphics import renderPDF

logger = logging.getLogger(__name__)


class ScoreCard(Flowable):
    """Visual scorecard component for financial metrics"""
    
    def __init__(self, width, height, score, max_score=10, label="Score", color_scheme="default"):
        self.width = width
        self.height = height
        self.score = score
        self.max_score = max_score
        self.label = label
        self.color_scheme = color_scheme
        
    def draw(self):
        # Background
        self.canv.setFillColor(colors.HexColor('#f8f9fa'))
        self.canv.rect(0, 0, self.width, self.height, fill=1, stroke=0)
        
        # Progress bar background
        bar_width = self.width * 0.6
        bar_height = 12
        bar_x = (self.width - bar_width) / 2
        bar_y = self.height * 0.3
        
        self.canv.setFillColor(colors.HexColor('#e9ecef'))
        self.canv.rect(bar_x, bar_y, bar_width, bar_height, fill=1, stroke=0)
        
        # Progress bar fill
        progress = min(self.score / self.max_score, 1.0)
        if progress >= 0.8:
            fill_color = colors.HexColor('#28a745')  # Green
        elif progress >= 0.6:
            fill_color = colors.HexColor('#ffc107')  # Yellow
        elif progress >= 0.4:
            fill_color = colors.HexColor('#fd7e14')  # Orange
        else:
            fill_color = colors.HexColor('#dc3545')  # Red
            
        self.canv.setFillColor(fill_color)
        self.canv.rect(bar_x, bar_y, bar_width * progress, bar_height, fill=1, stroke=0)
        
        # Score text
        self.canv.setFillColor(colors.black)
        self.canv.setFont("Helvetica-Bold", 16)
        self.canv.drawCentredText(self.width/2, self.height * 0.7, f"{self.score:.1f}/{self.max_score}")
        
        # Label
        self.canv.setFont("Helvetica", 10)
        self.canv.drawCentredText(self.width/2, self.height * 0.1, self.label)


class RecommendationBadge(Flowable):
    """Visual badge for investment recommendations"""
    
    def __init__(self, width, height, recommendation, confidence):
        self.width = width
        self.height = height
        self.recommendation = recommendation
        self.confidence = confidence
        
    def draw(self):
        # Badge color based on recommendation
        if self.recommendation in ['BUY', 'STRONG_BUY']:
            badge_color = colors.HexColor('#28a745')
        elif self.recommendation in ['SELL', 'STRONG_SELL']:
            badge_color = colors.HexColor('#dc3545')
        else:
            badge_color = colors.HexColor('#6c757d')
            
        # Draw badge background
        self.canv.setFillColor(badge_color)
        self.canv.roundRect(0, 0, self.width, self.height, 8, fill=1, stroke=0)
        
        # Recommendation text
        self.canv.setFillColor(colors.white)
        self.canv.setFont("Helvetica-Bold", 14)
        self.canv.drawCentredText(self.width/2, self.height * 0.6, self.recommendation)
        
        # Confidence text
        self.canv.setFont("Helvetica", 10)
        self.canv.drawCentredText(self.width/2, self.height * 0.25, f"{self.confidence} CONFIDENCE")


@dataclass
class ReportConfig:
    """Configuration for report generation"""
    title: str = "InvestiGator Investment Analysis"
    subtitle: str = "AI-Powered Investment Research Report"
    author: str = "InvestiGator AI System"
    include_charts: bool = True
    include_disclaimer: bool = True
    page_size: str = "letter"
    margin: float = 0.75 * inch


class NumberedCanvas(canvas.Canvas):
    """Custom canvas for page numbering"""
    def __init__(self, *args, **kwargs):
        canvas.Canvas.__init__(self, *args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        """Add page numbers to all pages"""
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_number(self, page_count):
        """Draw page number and disclaimer at bottom of page"""
        # Draw page number
        self.setFont("Helvetica", 9)
        self.drawRightString(
            self._pagesize[0] - 0.75 * inch,
            0.5 * inch,
            f"Page {self._pageNumber} of {page_count}"
        )
        
        # Draw disclaimer footer on every page
        self.setFont("Helvetica-Oblique", 8)
        self.setFillColor(colors.HexColor('#cc0000'))
        disclaimer_text = "AI-Generated Report - Educational Testing Only - NOT Investment Advice - See Full Disclaimer"
        self.drawCentredString(
            self._pagesize[0] / 2,
            0.5 * inch,
            disclaimer_text
        )
        self.setFillColor(colors.black)  # Reset color


class PDFReportGenerator:
    """Generates PDF investment reports"""
    
    def __init__(self, output_dir: Path, config: Optional[ReportConfig] = None):
        """
        Initialize PDF report generator
        
        Args:
            output_dir: Directory for output reports
            config: Report configuration
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or ReportConfig()
        
        # Initialize styles
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles"""
        # Helper function to safely add styles
        def safe_add_style(name, style):
            if name not in self.styles:
                self.styles.add(style)
        
        # Enhanced title style
        safe_add_style('CustomTitle', ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontSize=24,
            textColor=colors.HexColor('#1a1a1a'),
            spaceAfter=12,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))
        
        # Executive summary header
        safe_add_style('ExecutiveHeader', ParagraphStyle(
            name='ExecutiveHeader',
            parent=self.styles['Heading1'],
            fontSize=18,
            textColor=colors.HexColor('#0052cc'),
            spaceBefore=18,
            spaceAfter=12,
            fontName='Helvetica-Bold',
            borderWidth=2,
            borderColor=colors.HexColor('#0052cc'),
            borderPadding=8
        ))
        
        # Section header
        safe_add_style('SectionHeader', ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#2c3e50'),
            spaceBefore=12,
            spaceAfter=6,
            fontName='Helvetica-Bold',
            leftIndent=0,
            borderWidth=1,
            borderColor=colors.HexColor('#ecf0f1'),
            borderPadding=4
        ))
        
        # Highlight box style
        safe_add_style('HighlightBox', ParagraphStyle(
            name='HighlightBox',
            parent=self.styles['Normal'],
            fontSize=11,
            textColor=colors.HexColor('#2c3e50'),
            spaceBefore=6,
            spaceAfter=6,
            leftIndent=12,
            rightIndent=12,
            borderWidth=1,
            borderColor=colors.HexColor('#3498db'),
            borderPadding=8,
            backColor=colors.HexColor('#ebf3fd')
        ))
        
        # Risk warning style
        safe_add_style('RiskWarning', ParagraphStyle(
            name='RiskWarning',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=colors.HexColor('#c0392b'),
            spaceBefore=6,
            spaceAfter=6,
            leftIndent=12,
            rightIndent=12,
            borderWidth=1,
            borderColor=colors.HexColor('#e74c3c'),
            borderPadding=6,
            backColor=colors.HexColor('#fadbd8')
        ))
        
        # Metrics style
        safe_add_style('MetricsText', ParagraphStyle(
            name='MetricsText',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=colors.HexColor('#34495e'),
            spaceBefore=3,
            spaceAfter=3,
            leftIndent=6
        ))
        
        # Subtitle style
        safe_add_style('CustomSubtitle', ParagraphStyle(
            name='CustomSubtitle',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#444444'),
            spaceBefore=6,
            spaceAfter=12,
            alignment=TA_CENTER
        ))
        
        # Analysis text style
        safe_add_style('AnalysisText', ParagraphStyle(
            name='AnalysisText',
            parent=self.styles['BodyText'],
            fontSize=11,
            alignment=TA_JUSTIFY,
            spaceBefore=6,
            spaceAfter=6
        ))
    
    def generate_report(self, recommendations: List[Dict], 
                       report_type: str = "synthesis",
                       include_charts: Optional[List[str]] = None) -> str:
        """
        Generate PDF report from recommendations
        
        Args:
            recommendations: List of investment recommendations
            report_type: Type of report (synthesis, weekly, etc.)
            include_charts: List of chart paths to include
            
        Returns:
            Path to generated PDF report
        """
        # Create filename with symbol-based naming
        filename = self._generate_filename(recommendations, report_type)
        filepath = self.output_dir / filename
        
        # Create document
        doc = SimpleDocTemplate(
            str(filepath),
            pagesize=letter if self.config.page_size == "letter" else A4,
            rightMargin=self.config.margin,
            leftMargin=self.config.margin,
            topMargin=self.config.margin,
            bottomMargin=self.config.margin
        )
        
        # Build content
        story = []
        
        # Add title page
        story.extend(self._create_title_page(report_type))
        
        # Add executive summary
        story.extend(self._create_executive_summary(recommendations))
        
        # Add detailed analysis for each symbol
        for rec in recommendations:
            story.append(PageBreak())
            story.extend(self._create_symbol_analysis(rec, include_charts))
        
        # Add portfolio summary
        if len(recommendations) > 1:
            story.append(PageBreak())
            story.extend(self._create_portfolio_summary(recommendations))
        
        # Add charts section if provided
        if include_charts and self.config.include_charts:
            story.append(PageBreak())
            story.extend(self._create_charts_section(include_charts))
        
        # Add disclaimer
        if self.config.include_disclaimer:
            story.append(PageBreak())
            story.extend(self._create_disclaimer())
        
        # Build PDF with custom canvas for page numbers
        doc.build(story, canvasmaker=NumberedCanvas)
        
        logger.info(f"📄 Generated PDF report: {filepath}")
        return str(filepath)
    
    def _create_title_page(self, report_type: str) -> List:
        """Create title page with comprehensive legal disclaimer"""
        elements = []
        
        # Add title
        elements.append(Spacer(1, 0.5 * inch))
        elements.append(Paragraph(self.config.title, self.styles['CustomTitle']))
        elements.append(Paragraph(self.config.subtitle, self.styles['CustomSubtitle']))
        
        # Add report type
        report_type_text = report_type.replace('_', ' ').title()
        elements.append(Spacer(1, 0.3 * inch))
        elements.append(Paragraph(f"{report_type_text} Report", self.styles['Heading2']))
        
        # Add date
        elements.append(Spacer(1, 0.2 * inch))
        date_text = datetime.now().strftime('%B %d, %Y')
        elements.append(Paragraph(date_text, self.styles['Normal']))
        
        # Add comprehensive legal disclaimer
        elements.append(Spacer(1, 0.4 * inch))
        
        # Create prominent disclaimer style
        disclaimer_style = ParagraphStyle(
            'TitleDisclaimer',
            parent=self.styles['Normal'],
            fontSize=11,
            textColor=colors.HexColor('#cc0000'),
            borderWidth=2,
            borderColor=colors.HexColor('#cc0000'),
            borderPadding=10,
            borderRadius=4,
            backColor=colors.HexColor('#fff5f5'),
            spaceAfter=12,
            alignment=TA_CENTER
        )
        
        legal_disclaimer = """
        <b>IMPORTANT LEGAL DISCLAIMER</b><br/><br/>
        
        This report is generated entirely by artificial intelligence (AI) using Large Language Models (LLMs) 
        and is provided for <b>EDUCATIONAL and TESTING PURPOSES ONLY</b>.<br/><br/>
        
        <b>NOT INVESTMENT ADVICE:</b> The author is NOT a licensed investment advisor, financial planner, 
        broker-dealer, or any other financial professional. This report does NOT constitute investment advice, 
        financial advice, trading advice, or any other type of professional advice.<br/><br/>
        
        <b>AI-GENERATED CONTENT:</b> All analysis, recommendations, and insights in this report are 
        generated by AI systems which may contain errors, inaccuracies, hallucinations, or biases. 
        The AI has no fiduciary duty to you and cannot guarantee accuracy.<br/><br/>
        
        <b>NO WARRANTIES:</b> This report is provided "AS IS" without any warranties of any kind, 
        either express or implied, including but not limited to warranties of accuracy, completeness, 
        merchantability, or fitness for a particular purpose.<br/><br/>
        
        <b>USE AT YOUR OWN RISK:</b> Any investment decisions made based on this report are entirely at 
        your own risk. You could lose all of your invested capital. Past performance is not indicative 
        of future results. Always consult with qualified, licensed financial professionals before making 
        any investment decisions.<br/><br/>
        
        <b>NO LIABILITY:</b> The creators, developers, and operators of this AI system assume no liability 
        for any losses, damages, or consequences arising from the use of this report.
        """
        
        elements.append(Paragraph(legal_disclaimer, disclaimer_style))
        
        # Add author with clarification
        elements.append(Spacer(1, 0.3 * inch))
        elements.append(Paragraph(f"AI System: {self.config.author}", self.styles['Normal']))
        elements.append(Paragraph("<b>For Educational Testing Only - Not Professional Investment Advice</b>", 
                                self.styles['Normal']))
        
        return elements
    
    def _create_executive_summary(self, recommendations: List[Dict]) -> List:
        """Create enhanced executive summary with visual elements"""
        elements = []
        
        elements.append(Paragraph("Executive Summary", self.styles['ExecutiveHeader']))
        elements.append(Spacer(1, 0.3 * inch))
        
        # Portfolio Overview Section
        total_symbols = len(recommendations)
        buy_count = sum(1 for r in recommendations if 'BUY' in r.get('recommendation', '').upper())
        sell_count = sum(1 for r in recommendations if 'SELL' in r.get('recommendation', '').upper())
        hold_count = total_symbols - buy_count - sell_count
        
        avg_score = sum(r.get('overall_score', 5.0) for r in recommendations) / total_symbols if total_symbols > 0 else 0
        
        # Create portfolio overview table with visual elements
        overview_data = [
            ['Portfolio Snapshot', '', ''],
            ['Total Securities Analyzed', str(total_symbols), ''],
            ['Buy Recommendations', str(buy_count), f'{(buy_count/total_symbols*100):.0f}%' if total_symbols > 0 else '0%'],
            ['Hold Recommendations', str(hold_count), f'{(hold_count/total_symbols*100):.0f}%' if total_symbols > 0 else '0%'],
            ['Sell Recommendations', str(sell_count), f'{(sell_count/total_symbols*100):.0f}%' if total_symbols > 0 else '0%'],
            ['Average Investment Score', f'{avg_score:.1f}/10', self._get_score_rating(avg_score)]
        ]
        
        overview_table = Table(overview_data, colWidths=[2.5*inch, 1*inch, 1*inch])
        overview_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0052cc')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8f9fa')),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#dee2e6')),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')])
        ]))
        
        elements.append(overview_table)
        elements.append(Spacer(1, 0.3 * inch))
        
        # Top Recommendations Section with Visual Cards
        if recommendations:
            elements.append(Paragraph("Top Investment Opportunities", self.styles['SectionHeader']))
            elements.append(Spacer(1, 0.2 * inch))
            
            # Sort by overall score and get top 3
            top_recs = sorted(recommendations, key=lambda x: x.get('overall_score', 0), reverse=True)[:3]
            
            # Create visual cards for top recommendations
            for i, rec in enumerate(top_recs):
                symbol = rec.get('symbol', 'N/A')
                overall_score = rec.get('overall_score', 0)
                recommendation = rec.get('recommendation', 'N/A')
                confidence = rec.get('confidence', 'MEDIUM')
                current_price = rec.get('current_price', 0)
                price_target = rec.get('price_target', 0)
                
                # Create recommendation card data
                card_data = [
                    [f'#{i+1}', symbol, recommendation, f'{overall_score:.1f}/10'],
                    ['Price', f'${current_price:.2f}' if current_price else 'N/A', 
                     'Target', f'${price_target:.2f}' if price_target else 'N/A'],
                    ['Confidence', confidence, 'Upside', 
                     f'{((price_target/current_price-1)*100):.1f}%' if current_price and price_target else 'N/A']
                ]
                
                card_table = Table(card_data, colWidths=[0.5*inch, 1*inch, 1*inch, 1*inch])
                
                # Style based on recommendation
                if 'BUY' in recommendation.upper():
                    header_color = colors.HexColor('#28a745')
                elif 'SELL' in recommendation.upper():
                    header_color = colors.HexColor('#dc3545')
                else:
                    header_color = colors.HexColor('#6c757d')
                
                card_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), header_color),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 11),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8f9fa')),
                    ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#dee2e6')),
                    ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 1), (-1, -1), 9)
                ]))
                
                elements.append(card_table)
                elements.append(Spacer(1, 0.1 * inch))
        
        # Risk Assessment Summary
        elements.append(Spacer(1, 0.2 * inch))
        high_risk_count = sum(1 for r in recommendations if r.get('overall_score', 5) < 4)
        
        if high_risk_count > 0:
            risk_text = f"⚠️ <b>Risk Alert:</b> {high_risk_count} securities show elevated risk profiles. Review detailed analysis before investment decisions."
            elements.append(Paragraph(risk_text, self.styles['RiskWarning']))
        else:
            elements.append(Paragraph("✅ <b>Portfolio Risk:</b> All analyzed securities meet acceptable risk thresholds.", self.styles['HighlightBox']))
        
        return elements
    
    def _get_score_rating(self, score: float) -> str:
        """Convert numeric score to rating"""
        if score >= 8:
            return "Excellent"
        elif score >= 6:
            return "Good"
        elif score >= 4:
            return "Fair"
        else:
            return "Poor"
    
    def _create_symbol_analysis(self, recommendation: Dict, include_charts: Optional[List[str]] = None) -> List:
        """Create detailed analysis for a single symbol"""
        elements = []
        
        symbol = recommendation.get('symbol', 'N/A')
        
        # Header
        elements.append(Paragraph(f"{symbol} Analysis", self.styles['SectionHeader']))
        elements.append(Spacer(1, 0.1 * inch))
        
        # Get comprehensive analysis data first for enhanced scoring
        comprehensive_data = self._get_comprehensive_analysis_data(symbol)
        
        # Scores table - Enhanced with SEC comprehensive scores when available
        scores_data = [['Metric', 'Score', 'Rating']]
        
        # Core synthesis scores
        scores_data.extend([
            ['Overall Score', f"{recommendation.get('overall_score', 0):.1f}/10", 
             self._get_rating(recommendation.get('overall_score', 0))],
            ['Fundamental Score', f"{recommendation.get('fundamental_score', 0):.1f}/10",
             self._get_rating(recommendation.get('fundamental_score', 0))],
            ['Technical Score', f"{recommendation.get('technical_score', 0):.1f}/10",
             self._get_rating(recommendation.get('technical_score', 0))]
        ])
        
        # Financial statement component scores
        scores_data.extend([
            ['Income Statement', f"{recommendation.get('income_score', 0):.1f}/10",
             self._get_rating(recommendation.get('income_score', 0))],
            ['Cash Flow', f"{recommendation.get('cashflow_score', 0):.1f}/10",
             self._get_rating(recommendation.get('cashflow_score', 0))],
            ['Balance Sheet', f"{recommendation.get('balance_score', 0):.1f}/10",
             self._get_rating(recommendation.get('balance_score', 0))]
        ])
        
        # Investment characteristic scores
        scores_data.extend([
            ['Growth Score', f"{recommendation.get('growth_score', 0):.1f}/10",
             self._get_rating(recommendation.get('growth_score', 0))],
            ['Value Score', f"{recommendation.get('value_score', 0):.1f}/10",
             self._get_rating(recommendation.get('value_score', 0))]
        ])
        
        # Quality scores - prioritize comprehensive analysis when available, remove duplicates
        if comprehensive_data.get('business_quality_score') is not None:
            bq_score = comprehensive_data['business_quality_score']
            # Handle dict format for scores (e.g., {"score": 8.5, "explanation": "..."})
            if isinstance(bq_score, dict):
                bq_score = bq_score.get('score', 0)
            scores_data.append(['Business Quality', f"{float(bq_score):.1f}/10", self._get_rating(float(bq_score))])
        else:
            scores_data.append(['Business Quality', f"{recommendation.get('business_quality_score', 0):.1f}/10",
                               self._get_rating(recommendation.get('business_quality_score', 0))])
        
        if comprehensive_data.get('data_quality_score') is not None:
            dq_score = comprehensive_data['data_quality_score']
            # Handle dict format for scores
            if isinstance(dq_score, dict):
                dq_score = dq_score.get('score', 0)
            scores_data.append(['Data Quality', f"{float(dq_score):.1f}/10", self._get_rating(float(dq_score))])
        else:
            scores_data.append(['Data Quality', f"{recommendation.get('data_quality_score', 0):.1f}/10",
                               self._get_rating(recommendation.get('data_quality_score', 0))])
        
        scores_table = Table(scores_data, colWidths=[2.5*inch, 1.5*inch, 1.5*inch])
        scores_table.setStyle(TableStyle([
            # Header styling
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0052cc')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('TOPPADDING', (0, 0), (-1, 0), 8),
            
            # Body styling
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8f9fa')),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#dee2e6')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')]),
            
            # Score-based coloring for the Score column
            ('TEXTCOLOR', (1, 1), (1, -1), colors.black),
            ('TOPPADDING', (0, 1), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 6)
        ]))
        
        # Add conditional formatting for scores
        for i, row in enumerate(scores_data[1:], 1):  # Skip header
            try:
                score_text = row[1]
                if '/' in score_text:
                    score = float(score_text.split('/')[0])
                    if score >= 8:
                        scores_table.setStyle(TableStyle([('BACKGROUND', (1, i), (1, i), colors.HexColor('#d4edda'))]))  # Light green
                    elif score >= 6:
                        scores_table.setStyle(TableStyle([('BACKGROUND', (1, i), (1, i), colors.HexColor('#fff3cd'))]))  # Light yellow
                    elif score < 4:
                        scores_table.setStyle(TableStyle([('BACKGROUND', (1, i), (1, i), colors.HexColor('#f8d7da'))]))  # Light red
            except (ValueError, IndexError):
                pass
        
        elements.append(scores_table)
        elements.append(Spacer(1, 0.2 * inch))
        
        # Technical Analysis Summary (using structured data from direct extraction)
        tech_summary = self._create_technical_summary(recommendation)
        if tech_summary:
            elements.extend(tech_summary)
            elements.append(Spacer(1, 0.2 * inch))
        
        # Investment recommendation
        elements.append(Paragraph("<b>Investment Recommendation</b>", self.styles['Heading3']))
        rec_text = f"""
        <b>Recommendation:</b> {recommendation.get('recommendation', 'N/A')}<br/>
        <b>Confidence Level:</b> {recommendation.get('confidence', 'N/A')}<br/>
        <b>Time Horizon:</b> {recommendation.get('time_horizon', 'N/A')}<br/>
        <b>Position Size:</b> {recommendation.get('position_size', 'N/A')}<br/>
        """
        elements.append(Paragraph(rec_text, self.styles['AnalysisText']))
        
        # Price targets
        if recommendation.get('price_target'):
            elements.append(Spacer(1, 0.1 * inch))
            elements.append(Paragraph("<b>Price Analysis</b>", self.styles['Heading3']))
            price_text = f"""
            <b>Current Price:</b> ${recommendation.get('current_price', 0):.2f}<br/>
            <b>Price Target:</b> ${recommendation.get('price_target', 0):.2f}<br/>
            <b>Upside Potential:</b> {((recommendation.get('price_target', 0) / max(recommendation.get('current_price', 1), 0.01) - 1) * 100) if recommendation.get('current_price', 0) > 0 else 0:.1f}%<br/>
            <b>Stop Loss:</b> ${recommendation.get('stop_loss', 0):.2f}<br/>
            """
            elements.append(Paragraph(price_text, self.styles['AnalysisText']))
        
        # Investment thesis - prioritize SEC comprehensive analysis
        investment_thesis = self._get_comprehensive_investment_thesis(symbol, recommendation.get('investment_thesis', ''))
        if investment_thesis:
            elements.append(Spacer(1, 0.1 * inch))
            elements.append(Paragraph("<b>Investment Thesis</b>", self.styles['Heading3']))
            # Convert markdown to HTML for proper rendering
            investment_thesis_html = self._markdown_to_html(investment_thesis)
            elements.append(Paragraph(investment_thesis_html, self.styles['AnalysisText']))
        
        # Key insights - prioritize SEC comprehensive analysis
        insights_to_show = comprehensive_data.get('key_insights') or recommendation.get('key_insights', [])
        insights_source = "SEC Comprehensive" if comprehensive_data.get('key_insights') else "Synthesis"
        
        if insights_to_show:
            elements.append(Spacer(1, 0.1 * inch))
            elements.append(Paragraph(f"<b>Key Insights ({insights_source})</b>", self.styles['Heading3']))
            for insight in insights_to_show[:5]:  # Show top 5 insights
                elements.append(Paragraph(f"• {insight}", self.styles['AnalysisText']))
        
        # Key risks - prioritize SEC comprehensive analysis
        risks_to_show = comprehensive_data.get('key_risks') or recommendation.get('key_risks', [])
        risks_source = "SEC Comprehensive" if comprehensive_data.get('key_risks') else "Synthesis"
        
        if risks_to_show:
            elements.append(Spacer(1, 0.1 * inch))
            elements.append(Paragraph(f"<b>Key Risks ({risks_source})</b>", self.styles['Heading3']))
            for risk in risks_to_show[:5]:  # Show top 5 risks
                elements.append(Paragraph(f"• {risk}", self.styles['AnalysisText']))
        
        # SEC Trend Analysis (if available)
        if comprehensive_data.get('trend_analysis'):
            trend_analysis = comprehensive_data['trend_analysis']
            elements.append(Spacer(1, 0.1 * inch))
            elements.append(Paragraph("<b>Trend Analysis (SEC)</b>", self.styles['Heading3']))
            
            trend_items = []
            if trend_analysis.get('revenue_trend'):
                trend_items.append(f"Revenue Trend: {trend_analysis['revenue_trend']}")
            if trend_analysis.get('margin_trend'):
                trend_items.append(f"Margin Trend: {trend_analysis['margin_trend']}")
            if trend_analysis.get('cash_flow_trend'):
                trend_items.append(f"Cash Flow Trend: {trend_analysis['cash_flow_trend']}")
            
            for item in trend_items:
                elements.append(Paragraph(f"• {item}", self.styles['AnalysisText']))
        
        # LLM Thinking and Details Sections
        sec_thinking, tech_thinking = self._get_llm_thinking_details(symbol)
        
        # SEC Fundamental Analysis Thinking
        if sec_thinking:
            elements.append(Spacer(1, 0.1 * inch))
            elements.append(Paragraph("<b>SEC Fundamental Analysis Thinking</b>", self.styles['Heading3']))
            # Convert markdown to HTML for proper rendering
            sec_thinking_html = self._markdown_to_html(sec_thinking)
            elements.append(Paragraph(sec_thinking_html, self.styles['AnalysisText']))
        
        # Technical Analysis Thinking  
        if tech_thinking:
            elements.append(Spacer(1, 0.1 * inch))
            elements.append(Paragraph("<b>Technical Analysis Thinking</b>", self.styles['Heading3']))
            # Convert markdown to HTML for proper rendering
            tech_thinking_html = self._markdown_to_html(tech_thinking)
            elements.append(Paragraph(tech_thinking_html, self.styles['AnalysisText']))
        
        # Analysis thinking process
        if recommendation.get('analysis_thinking'):
            elements.append(Spacer(1, 0.1 * inch))
            elements.append(Paragraph("<b>Synthesis Analysis Reasoning</b>", self.styles['Heading3']))
            # Convert markdown to HTML for proper rendering
            analysis_thinking_html = self._markdown_to_html(recommendation.get('analysis_thinking', ''))
            elements.append(Paragraph(analysis_thinking_html, self.styles['AnalysisText']))
        
        # Synthesis details
        if recommendation.get('synthesis_details'):
            elements.append(Spacer(1, 0.1 * inch))
            elements.append(Paragraph("<b>Synthesis Methodology</b>", self.styles['Heading3']))
            # Convert markdown to HTML for proper rendering
            synthesis_details_html = self._markdown_to_html(recommendation.get('synthesis_details', ''))
            elements.append(Paragraph(synthesis_details_html, self.styles['AnalysisText']))
        
        # Add technical chart if available
        if include_charts and self.config.include_charts:
            tech_chart = f"{symbol}_technical_analysis.png"
            for chart_path in include_charts:
                if tech_chart in chart_path and Path(chart_path).exists():
                    elements.append(Spacer(1, 0.2 * inch))
                    elements.append(Paragraph("<b>Technical Analysis Chart</b>", self.styles['Heading3']))
                    img = Image(chart_path, width=6*inch, height=4*inch)
                    elements.append(img)
                    break
        
        return elements
    
    def _create_portfolio_summary(self, recommendations: List[Dict]) -> List:
        """Create portfolio summary section"""
        elements = []
        
        elements.append(Paragraph("Portfolio Summary", self.styles['SectionHeader']))
        elements.append(Spacer(1, 0.2 * inch))
        
        # Sort by score
        sorted_recs = sorted(recommendations, key=lambda x: x.get('overall_score', 0), reverse=True)
        
        # Create summary table
        table_data = [['Symbol', 'Recommendation', 'Overall Score', 'Target Return', 'Position Size']]
        
        for rec in sorted_recs:
            symbol = rec.get('symbol', 'N/A')
            recommendation = rec.get('recommendation', 'N/A')
            score = f"{rec.get('overall_score', 0):.1f}"
            
            # Calculate target return
            current = rec.get('current_price', 0) or 0
            target = rec.get('price_target', 0) or 0
            target_return = ((target / current - 1) * 100) if current and current > 0 and target and target > 0 else 0
            target_return_str = f"{target_return:+.1f}%" if target and target > 0 and current and current > 0 else "N/A"
            
            position = rec.get('position_size', 'N/A')
            
            table_data.append([symbol, recommendation, score, target_return_str, position])
        
        # Create table
        summary_table = Table(table_data, colWidths=[1.2*inch, 1.8*inch, 1.2*inch, 1.3*inch, 1.5*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 10)
        ]))
        
        elements.append(summary_table)
        
        return elements
    
    def _create_charts_section(self, chart_paths: List[str]) -> List:
        """Create section with additional charts"""
        elements = []
        
        elements.append(Paragraph("Analysis Charts", self.styles['SectionHeader']))
        elements.append(Spacer(1, 0.2 * inch))
        
        # Add 3D fundamental chart
        for chart_path in chart_paths:
            if '3d_fundamental' in chart_path and Path(chart_path).exists():
                elements.append(Paragraph("<b>3D Fundamental Analysis</b>", self.styles['Heading3']))
                img = Image(chart_path, width=6*inch, height=4.5*inch)
                elements.append(img)
                elements.append(Spacer(1, 0.3 * inch))
                break
        
        # Add 2D technical vs fundamental chart
        for chart_path in chart_paths:
            if '2d_technical_fundamental' in chart_path and Path(chart_path).exists():
                elements.append(Paragraph("<b>Technical vs Fundamental Analysis</b>", self.styles['Heading3']))
                img = Image(chart_path, width=6*inch, height=4.5*inch)
                elements.append(img)
                elements.append(Spacer(1, 0.3 * inch))
                break
        
        # Add growth vs value chart
        for chart_path in chart_paths:
            if 'growth_value' in chart_path and Path(chart_path).exists():
                elements.append(Paragraph("<b>Growth vs Value Positioning</b>", self.styles['Heading3']))
                img = Image(chart_path, width=6*inch, height=4.5*inch)
                elements.append(img)
                break
        
        return elements
    
    def _create_disclaimer(self) -> List:
        """Create disclaimer section"""
        elements = []
        
        elements.append(Paragraph("Full Legal Disclaimer", self.styles['SectionHeader']))
        elements.append(Spacer(1, 0.1 * inch))
        
        disclaimer_text = """
        <b>AI-GENERATED REPORT - NOT INVESTMENT ADVICE</b><br/><br/>
        
        This report is generated entirely by artificial intelligence using Large Language Models (LLMs) and is 
        provided for <b>educational and testing purposes only</b>. The creators are NOT licensed investment advisors, 
        broker-dealers, or financial professionals.<br/><br/>
        
        <b>IMPORTANT WARNINGS:</b><br/>
        • This is NOT investment advice or a recommendation to buy, sell, or hold any securities<br/>
        • All content is AI-generated and may contain errors, inaccuracies, or hallucinations<br/>
        • Past performance is not indicative of future results<br/>
        • You could lose all invested capital - investments carry substantial risk<br/>
        • The AI system has no fiduciary duty and cannot guarantee accuracy<br/><br/>
        
        <b>REQUIRED ACTIONS BEFORE INVESTING:</b><br/>
        • Conduct your own thorough research and due diligence<br/>
        • Consult with licensed, qualified financial advisors<br/>
        • Verify all information independently<br/>
        • Consider your personal financial situation and risk tolerance<br/><br/>
        
        <b>NO LIABILITY:</b> The creators, developers, and operators of InvestiGator assume no liability whatsoever 
        for any losses, damages, or consequences arising from the use of this report. Use of this report is entirely 
        at your own risk.<br/><br/>
        
        <b>REGULATORY NOTICE:</b> This report has not been reviewed or approved by any regulatory authority. 
        It is not intended for distribution in jurisdictions where such distribution would be unlawful.<br/><br/>
        
        Generated by InvestiGator - AI-Powered Investment Research System<br/>
        For Educational Testing Only - Not Professional Investment Advice
        """
        
        elements.append(Paragraph(disclaimer_text, self.styles['AnalysisText']))
        
        return elements
    
    def _generate_filename(self, recommendations: List[Dict], report_type: str) -> str:
        """
        Generate filename based on symbols in recommendations
        
        Args:
            recommendations: List of investment recommendations
            report_type: Type of report
            
        Returns:
            Generated filename
        """
        if not recommendations:
            return f"{report_type}_report_no_symbols.pdf"
        
        # Extract symbols from recommendations
        symbols = [rec.get('symbol', 'UNKNOWN') for rec in recommendations]
        
        # Create symbol part of filename
        if len(symbols) <= 4:
            # Use all symbols separated by hyphens
            symbol_part = '-'.join(symbols)
        elif len(symbols) == 5:
            # Use all 5 symbols
            symbol_part = '-'.join(symbols)
        else:
            # Use first 4 symbols + "OTHERS"
            symbol_part = '-'.join(symbols[:4]) + '-OTHERS'
        
        # Create final filename
        filename = f"{report_type}_report_{symbol_part}.pdf"
        
        return filename
    
    def _get_comprehensive_analysis_data(self, symbol: str) -> dict:
        """
        Get comprehensive analysis data from SEC comprehensive analysis cache
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary containing all comprehensive analysis data including scores and insights
        """
        try:
            # Import cache manager and types
            from utils.cache.cache_manager import CacheManager
            from utils.cache.cache_types import CacheType
            from config import get_config
            
            # Initialize cache manager
            config = get_config()
            cache_manager = CacheManager(config)
            
            # Determine the most recent fiscal period (use current assumptions)
            from datetime import datetime
            current_date = datetime.now()
            current_year = current_date.year
            
            # Try to find SEC comprehensive analysis data
            # First try with current year comprehensive
            cache_key = {
                'symbol': symbol,
                'form_type': 'COMPREHENSIVE',
                'period': f"{current_year}-FY",
                'llm_type': 'sec'
            }
            
            cached_response = cache_manager.get(CacheType.LLM_RESPONSE, cache_key)
            
            # If not found, try previous year
            if not cached_response:
                cache_key['period'] = f"{current_year - 1}-FY"
                cached_response = cache_manager.get(CacheType.LLM_RESPONSE, cache_key)
            
            if cached_response:
                # Parse the comprehensive analysis response
                response_content = cached_response.get('response', cached_response.get('content', ''))
                
                if response_content:
                    try:
                        # Try to parse as JSON
                        import json
                        if isinstance(response_content, str):
                            comp_analysis = json.loads(response_content)
                        else:
                            comp_analysis = response_content
                        
                        # Extract all valuable data
                        comprehensive_data = {
                            'financial_health_score': comp_analysis.get('financial_health_score'),
                            'business_quality_score': comp_analysis.get('business_quality_score'),
                            'growth_prospects_score': comp_analysis.get('growth_prospects_score'),
                            'data_quality_score': comp_analysis.get('data_quality_score'),
                            'overall_score': comp_analysis.get('overall_score'),
                            'trend_analysis': comp_analysis.get('trend_analysis', {}),
                            'key_insights': comp_analysis.get('key_insights', []),
                            'key_risks': comp_analysis.get('key_risks', []),
                            'investment_thesis': comp_analysis.get('investment_thesis', ''),
                            'confidence_level': comp_analysis.get('confidence_level', ''),
                            'analysis_summary': comp_analysis.get('analysis_summary', ''),
                            'quarterly_analyses': comp_analysis.get('quarterly_analyses', []),
                            'quarters_analyzed': comp_analysis.get('quarters_analyzed', 0)
                        }
                        
                        logger.info(f"📊 Extracted comprehensive analysis data for {symbol}")
                        return comprehensive_data
                        
                    except (json.JSONDecodeError, AttributeError, KeyError) as e:
                        logger.warning(f"Failed to parse SEC comprehensive analysis for {symbol}: {e}")
            
            # Return empty dict if no data found
            return {}
            
        except Exception as e:
            logger.error(f"Error extracting comprehensive analysis data for {symbol}: {e}")
            return {}

    def _get_comprehensive_investment_thesis(self, symbol: str, fallback_thesis: str = '') -> str:
        """
        Get investment thesis from SEC comprehensive analysis cache
        
        Args:
            symbol: Stock symbol
            fallback_thesis: Fallback thesis from synthesis if SEC data unavailable
            
        Returns:
            Investment thesis text from SEC comprehensive analysis or fallback
        """
        try:
            # Get comprehensive analysis data using new extraction method
            comp_data = self._get_comprehensive_analysis_data(symbol)
            
            if comp_data and comp_data.get('investment_thesis'):
                logger.info(f"📊 Using SEC comprehensive investment thesis for {symbol}")
                return comp_data['investment_thesis']
            
            # Try to build thesis from key insights and risks
            if comp_data and (comp_data.get('key_insights') or comp_data.get('key_risks')):
                thesis_parts = []
                
                if comp_data.get('key_insights'):
                    thesis_parts.append("Key Investment Insights:")
                    for insight in comp_data['key_insights'][:3]:  # Top 3 insights
                        thesis_parts.append(f"• {insight}")
                
                if comp_data.get('key_risks'):
                    thesis_parts.append("\nKey Risk Factors:")
                    for risk in comp_data['key_risks'][:2]:  # Top 2 risks
                        thesis_parts.append(f"• {risk}")
                
                # Add analysis summary if available
                if comp_data.get('analysis_summary'):
                    thesis_parts.append(f"\nOverall Assessment: {comp_data['analysis_summary']}")
                
                if thesis_parts:
                    comprehensive_thesis = "\n".join(thesis_parts)
                    logger.info(f"📊 Built comprehensive investment thesis for {symbol} from insights")
                    return comprehensive_thesis
            
            # Fallback to synthesis thesis if SEC data unavailable
            if fallback_thesis:
                logger.info(f"📈 Using synthesis investment thesis for {symbol} (SEC comprehensive not available)")
                return fallback_thesis
            
            # Ultimate fallback
            logger.warning(f"No investment thesis available for {symbol}")
            return f"Investment analysis for {symbol} based on fundamental and technical factors."
            
        except Exception as e:
            logger.error(f"Error fetching comprehensive investment thesis for {symbol}: {e}")
            return fallback_thesis or f"Investment analysis for {symbol} based on available data."
    
    def _create_technical_summary(self, recommendation: Dict) -> List:
        """Create visual technical analysis summary using structured data"""
        elements = []
        
        # Extract technical indicators from the recommendation (set by direct extraction)
        support_levels = recommendation.get('support_levels', [])
        resistance_levels = recommendation.get('resistance_levels', [])
        trend_direction = recommendation.get('trend_direction', 'NEUTRAL')
        technical_score = recommendation.get('technical_score', 0)
        
        # Only create section if we have technical data
        if not (support_levels or resistance_levels or trend_direction != 'NEUTRAL'):
            return elements
            
        elements.append(Paragraph("Technical Analysis Summary", self.styles['SectionHeader']))
        
        # Technical overview table
        tech_data = [['Technical Metric', 'Value', 'Interpretation']]
        
        # Add technical score
        if technical_score > 0:
            tech_data.append(['Technical Score', f'{technical_score:.1f}/10', self._get_rating(technical_score)])
        
        # Add trend information
        if trend_direction and trend_direction != 'NEUTRAL':
            trend_color = '🟢' if trend_direction == 'BULLISH' else '🔴' if trend_direction == 'BEARISH' else '🟡'
            tech_data.append(['Trend Direction', f'{trend_color} {trend_direction}', 
                             'Favorable' if trend_direction == 'BULLISH' else 'Concerning' if trend_direction == 'BEARISH' else 'Neutral'])
        
        # Add support levels
        if support_levels:
            support_str = ', '.join([f'${level:.2f}' for level in support_levels[:3]])
            tech_data.append(['Key Support Levels', support_str, 'Downside protection'])
        
        # Add resistance levels
        if resistance_levels:
            resistance_str = ', '.join([f'${level:.2f}' for level in resistance_levels[:3]])
            tech_data.append(['Key Resistance Levels', resistance_str, 'Upside targets'])
        
        # Create table if we have data
        if len(tech_data) > 1:
            tech_table = Table(tech_data, colWidths=[2*inch, 2*inch, 2*inch])
            tech_table.setStyle(TableStyle([
                # Header styling
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#17a2b8')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 11),
                
                # Body styling
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8f9fa')),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#dee2e6')),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')]),
                ('TOPPADDING', (0, 1), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 1), (-1, -1), 4)
            ]))
            
            elements.append(tech_table)
            
            # Add technical insights if available
            momentum_signals = recommendation.get('momentum_signals', [])
            if momentum_signals:
                elements.append(Spacer(1, 0.1 * inch))
                elements.append(Paragraph("<b>Technical Signals</b>", self.styles['Heading3']))
                for signal in momentum_signals[:3]:  # Show top 3 signals
                    elements.append(Paragraph(f"• {signal}", self.styles['MetricsText']))
        
        return elements
    
    def _get_rating(self, score: float) -> str:
        """Get rating text from score"""
        if score >= 8:
            return "Excellent"
        elif score >= 6:
            return "Good"
        elif score >= 4:
            return "Fair"
        else:
            return "Poor"
    
    def _markdown_to_html(self, markdown_text: str) -> str:
        """
        Convert markdown text to HTML suitable for ReportLab Paragraph
        
        Args:
            markdown_text: Text with markdown formatting
            
        Returns:
            HTML-formatted text suitable for ReportLab
        """
        if not markdown_text:
            return ""
        
        try:
            # Convert markdown to HTML
            html = markdown.markdown(markdown_text, extensions=['nl2br'])
            
            # Clean up HTML for ReportLab compatibility
            # ReportLab uses a limited subset of HTML tags
            
            # Replace markdown bold (**text**) with HTML bold
            html = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', html)
            
            # Replace markdown italic (*text*) with HTML italic  
            html = re.sub(r'\*(.*?)\*', r'<i>\1</i>', html)
            
            # Ensure bullet points are properly formatted
            html = re.sub(r'<li>(.*?)</li>', r'• \1<br/>', html)
            
            # Remove unsupported HTML tags but keep content
            html = re.sub(r'</?(?:ul|ol)>', '', html)
            html = re.sub(r'</?p>', '', html)
            
            # Convert line breaks
            html = html.replace('\n', '<br/>')
            
            # Clean up multiple consecutive line breaks
            html = re.sub(r'(<br/>){3,}', '<br/><br/>', html)
            
            return html.strip()
            
        except Exception as e:
            logger.warning(f"Failed to convert markdown to HTML: {e}")
            # Fallback: basic markdown conversion
            text = markdown_text
            text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
            text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
            text = text.replace('\n', '<br/>')
            return text
    
    def _get_llm_thinking_details(self, symbol: str) -> tuple:
        """
        Extract thinking and details from SEC fundamental and technical analysis LLM responses
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Tuple of (sec_thinking, tech_thinking) strings
        """
        try:
            # Import cache manager and types
            from utils.cache.cache_manager import CacheManager
            from utils.cache.cache_types import CacheType
            from config import get_config
            
            # Initialize cache manager
            config = get_config()
            cache_manager = CacheManager(config)
            
            sec_thinking = ""
            tech_thinking = ""
            
            # Get SEC comprehensive analysis thinking
            from datetime import datetime
            current_date = datetime.now()
            current_year = current_date.year
            
            # Try to find SEC comprehensive analysis response
            cache_key = {
                'symbol': symbol,
                'form_type': 'COMPREHENSIVE',
                'period': f"{current_year}-FY",
                'llm_type': 'sec'
            }
            
            cached_response = cache_manager.get(CacheType.LLM_RESPONSE, cache_key)
            
            # If not found, try previous year
            if not cached_response:
                cache_key['period'] = f"{current_year - 1}-FY"
                cached_response = cache_manager.get(CacheType.LLM_RESPONSE, cache_key)
            
            if cached_response:
                response_content = cached_response.get('response', cached_response.get('content', ''))
                
                if response_content:
                    try:
                        # Try to parse as JSON and extract thinking/details
                        import json
                        if isinstance(response_content, str):
                            comp_analysis = json.loads(response_content)
                        else:
                            comp_analysis = response_content
                        
                        # Extract thinking/reasoning fields
                        thinking_fields = []
                        
                        # Extract analysis summary 
                        if comp_analysis.get('analysis_summary'):
                            thinking_fields.append(f"**Analysis Summary**: {comp_analysis['analysis_summary']}")
                        
                        # Extract investment thesis
                        if comp_analysis.get('investment_thesis'):
                            thinking_fields.append(f"**Investment Thesis**: {comp_analysis['investment_thesis']}")
                        
                        # Extract top-level detail field (contextual summary)
                        if comp_analysis.get('detail'):
                            thinking_fields.append(f"**Detailed Analysis**: {comp_analysis['detail']}")
                        
                        # Extract quarterly analysis details
                        quarterly_analyses = comp_analysis.get('quarterly_analyses', [])
                        if quarterly_analyses and len(quarterly_analyses) > 0:
                            # Get the most recent quarter's detailed analysis
                            recent_quarter = quarterly_analyses[0]
                            if recent_quarter.get('detail'):
                                thinking_fields.append(f"**Recent Quarter Details**: {recent_quarter['detail']}")
                        
                        if thinking_fields:
                            sec_thinking = "\n\n".join(thinking_fields)
                            
                    except (json.JSONDecodeError, AttributeError, KeyError) as e:
                        logger.warning(f"Failed to parse SEC thinking for {symbol}: {e}")
            
            # Get technical analysis thinking
            tech_cache_key = {
                'symbol': symbol,
                'analysis_type': 'technical_indicators',
                'llm_type': 'technical'
            }
            
            tech_cached_response = cache_manager.get(CacheType.LLM_RESPONSE, tech_cache_key)
            
            # Fallback: Check file-based technical analysis cache
            if not tech_cached_response:
                try:
                    from pathlib import Path
                    tech_file_path = f"data/llm_cache/{symbol}/response_technical_indicators.txt"
                    if Path(tech_file_path).exists():
                        with open(tech_file_path, 'r') as f:
                            tech_file_content = f.read()
                        tech_cached_response = {
                            'response': tech_file_content,
                            'content': tech_file_content,
                            'metadata': {'source': 'file_fallback'}
                        }
                        logger.info(f"📊 Using file fallback for technical analysis thinking: {symbol}")
                except Exception as e:
                    logger.warning(f"Failed to read technical analysis file for {symbol}: {e}")
            
            if tech_cached_response:
                tech_content = tech_cached_response.get('response', tech_cached_response.get('content', ''))
                
                if tech_content:
                    try:
                        # Handle file format with headers - extract JSON part
                        json_content = tech_content
                        if "=== AI RESPONSE ===" in tech_content:
                            json_start = tech_content.find("=== AI RESPONSE ===") + len("=== AI RESPONSE ===")
                            json_content = tech_content[json_start:].strip()
                        
                        # Try to parse as JSON and extract thinking/details
                        if isinstance(json_content, str):
                            tech_analysis = json.loads(json_content)
                        else:
                            tech_analysis = json_content
                        
                        # Extract thinking/reasoning fields
                        tech_thinking_fields = []
                        
                        if tech_analysis.get('thinking'):
                            # Clean up the thinking text (remove escaped newlines)
                            thinking_text = tech_analysis['thinking'].replace('\\n', '\n').strip()
                            tech_thinking_fields.append(f"**Technical Analysis Process**: {thinking_text}")
                        
                        # Extract top-level detail field (contextual technical summary)
                        if tech_analysis.get('detail'):
                            detail_text = tech_analysis['detail'].replace('\\n', '\n').strip()
                            tech_thinking_fields.append(f"**Technical Detail Summary**: {detail_text}")
                        
                        # Add technical signals summary
                        if tech_analysis.get('momentum_signals'):
                            signals_text = "\n".join([f"• {signal}" for signal in tech_analysis['momentum_signals']])
                            tech_thinking_fields.append(f"**Key Technical Signals**:\n{signals_text}")
                        
                        # Add risk factors
                        if tech_analysis.get('risk_factors'):
                            risks_text = "\n".join([f"• {risk}" for risk in tech_analysis['risk_factors']])
                            tech_thinking_fields.append(f"**Technical Risk Factors**:\n{risks_text}")
                        
                        if tech_thinking_fields:
                            tech_thinking = "\n\n".join(tech_thinking_fields)
                            
                    except (json.JSONDecodeError, AttributeError, KeyError) as e:
                        logger.warning(f"Failed to parse technical thinking for {symbol}: {e}")
            
            return (sec_thinking, tech_thinking)
            
        except Exception as e:
            logger.error(f"Error extracting LLM thinking for {symbol}: {e}")
            return ("", "")