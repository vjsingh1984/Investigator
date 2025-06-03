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

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, 
    PageBreak, Image, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.pdfgen import canvas
from reportlab.platypus.flowables import Flowable

logger = logging.getLogger(__name__)


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
        """Draw page number at bottom of page"""
        self.setFont("Helvetica", 9)
        self.drawRightString(
            self._pagesize[0] - 0.75 * inch,
            0.5 * inch,
            f"Page {self._pageNumber} of {page_count}"
        )


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
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontSize=24,
            textColor=colors.HexColor('#1a1a1a'),
            spaceAfter=12,
            alignment=TA_CENTER
        ))
        
        # Subtitle style
        self.styles.add(ParagraphStyle(
            name='CustomSubtitle',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#444444'),
            spaceBefore=6,
            spaceAfter=12,
            alignment=TA_CENTER
        ))
        
        # Section header style
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading1'],
            fontSize=18,
            textColor=colors.HexColor('#2c3e50'),
            spaceBefore=12,
            spaceAfter=6,
            borderWidth=1,
            borderColor=colors.HexColor('#2c3e50'),
            borderPadding=4
        ))
        
        # Analysis text style
        self.styles.add(ParagraphStyle(
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
        """Create title page elements"""
        elements = []
        
        # Add title
        elements.append(Spacer(1, 2 * inch))
        elements.append(Paragraph(self.config.title, self.styles['CustomTitle']))
        elements.append(Paragraph(self.config.subtitle, self.styles['CustomSubtitle']))
        
        # Add report type
        report_type_text = report_type.replace('_', ' ').title()
        elements.append(Spacer(1, 0.5 * inch))
        elements.append(Paragraph(f"{report_type_text} Report", self.styles['Heading2']))
        
        # Add date
        elements.append(Spacer(1, 0.5 * inch))
        date_text = datetime.now().strftime('%B %d, %Y')
        elements.append(Paragraph(date_text, self.styles['Normal']))
        
        # Add author
        elements.append(Spacer(1, 2 * inch))
        elements.append(Paragraph(f"Prepared by: {self.config.author}", self.styles['Normal']))
        
        return elements
    
    def _create_executive_summary(self, recommendations: List[Dict]) -> List:
        """Create executive summary section"""
        elements = []
        
        elements.append(Paragraph("Executive Summary", self.styles['SectionHeader']))
        elements.append(Spacer(1, 0.2 * inch))
        
        # Summary statistics
        total_symbols = len(recommendations)
        buy_count = sum(1 for r in recommendations if 'BUY' in r.get('recommendation', '').upper())
        sell_count = sum(1 for r in recommendations if 'SELL' in r.get('recommendation', '').upper())
        hold_count = total_symbols - buy_count - sell_count
        
        avg_score = sum(r.get('overall_score', 5.0) for r in recommendations) / total_symbols if total_symbols > 0 else 0
        
        summary_text = f"""
        This report analyzes {total_symbols} securities using advanced AI-powered fundamental and technical analysis.
        
        <b>Portfolio Overview:</b><br/>
        • Total Securities Analyzed: {total_symbols}<br/>
        • Buy Recommendations: {buy_count}<br/>
        • Hold Recommendations: {hold_count}<br/>
        • Sell Recommendations: {sell_count}<br/>
        • Average Investment Score: {avg_score:.1f}/10<br/>
        
        <b>Key Findings:</b><br/>
        """
        
        elements.append(Paragraph(summary_text, self.styles['AnalysisText']))
        
        # Add top recommendations
        top_recs = sorted(recommendations, key=lambda x: x.get('overall_score', 0), reverse=True)[:3]
        for rec in top_recs:
            symbol = rec.get('symbol', 'N/A')
            score = rec.get('overall_score', 0)
            recommendation = rec.get('recommendation', 'N/A')
            elements.append(Paragraph(
                f"• <b>{symbol}</b>: {recommendation} (Score: {score:.1f}/10)",
                self.styles['AnalysisText']
            ))
        
        return elements
    
    def _create_symbol_analysis(self, recommendation: Dict, include_charts: Optional[List[str]] = None) -> List:
        """Create detailed analysis for a single symbol"""
        elements = []
        
        symbol = recommendation.get('symbol', 'N/A')
        
        # Header
        elements.append(Paragraph(f"{symbol} Analysis", self.styles['SectionHeader']))
        elements.append(Spacer(1, 0.1 * inch))
        
        # Scores table
        scores_data = [
            ['Metric', 'Score', 'Rating'],
            ['Overall Score', f"{recommendation.get('overall_score', 0):.1f}/10", 
             self._get_rating(recommendation.get('overall_score', 0))],
            ['Fundamental Score', f"{recommendation.get('fundamental_score', 0):.1f}/10",
             self._get_rating(recommendation.get('fundamental_score', 0))],
            ['Technical Score', f"{recommendation.get('technical_score', 0):.1f}/10",
             self._get_rating(recommendation.get('technical_score', 0))],
            ['Income Statement', f"{recommendation.get('income_score', 0):.1f}/10",
             self._get_rating(recommendation.get('income_score', 0))],
            ['Cash Flow', f"{recommendation.get('cashflow_score', 0):.1f}/10",
             self._get_rating(recommendation.get('cashflow_score', 0))],
            ['Balance Sheet', f"{recommendation.get('balance_score', 0):.1f}/10",
             self._get_rating(recommendation.get('balance_score', 0))]
        ]
        
        scores_table = Table(scores_data, colWidths=[2.5*inch, 1.5*inch, 1.5*inch])
        scores_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        elements.append(scores_table)
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
        
        # Investment thesis
        if recommendation.get('investment_thesis'):
            elements.append(Spacer(1, 0.1 * inch))
            elements.append(Paragraph("<b>Investment Thesis</b>", self.styles['Heading3']))
            elements.append(Paragraph(recommendation.get('investment_thesis', ''), self.styles['AnalysisText']))
        
        # Key insights
        if recommendation.get('key_insights'):
            elements.append(Spacer(1, 0.1 * inch))
            elements.append(Paragraph("<b>Key Insights</b>", self.styles['Heading3']))
            for insight in recommendation.get('key_insights', []):
                elements.append(Paragraph(f"• {insight}", self.styles['AnalysisText']))
        
        # Key risks
        if recommendation.get('key_risks'):
            elements.append(Spacer(1, 0.1 * inch))
            elements.append(Paragraph("<b>Key Risks</b>", self.styles['Heading3']))
            for risk in recommendation.get('key_risks', []):
                elements.append(Paragraph(f"• {risk}", self.styles['AnalysisText']))
        
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
                break
        
        return elements
    
    def _create_disclaimer(self) -> List:
        """Create disclaimer section"""
        elements = []
        
        elements.append(Paragraph("Disclaimer", self.styles['SectionHeader']))
        elements.append(Spacer(1, 0.1 * inch))
        
        disclaimer_text = """
        This report is generated by an AI-powered system and is provided for informational purposes only. 
        It does not constitute investment advice, and should not be relied upon as a basis for making investment decisions. 
        
        The information contained in this report is based on publicly available data and AI analysis, which may contain errors or omissions. 
        Past performance is not indicative of future results. All investments carry risk, including the potential loss of principal.
        
        Always conduct your own research and consult with qualified financial advisors before making investment decisions.
        The creators of this report assume no liability for any losses or damages arising from the use of this information.
        
        Generated by InvestiGator - AI-Powered Investment Research System
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