#!/usr/bin/env python3
"""
InvestiGator - Peer Group PDF Report Generation Module
Copyright (c) 2025 Vijaykumar Singh
Licensed under the Apache License 2.0

Specialized PDF Report Generation for Peer Group Analysis
Handles creation of comprehensive peer group investment reports
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass
import json
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

from utils.chart_generator import ChartGenerator
from patterns.analysis.peer_comparison import get_peer_comparison_analyzer
from utils.cache.cache_manager import CacheManager
from utils.cache.cache_types import CacheType
import statistics
import re

logger = logging.getLogger(__name__)


@dataclass
class PeerGroupReportConfig:
    """Configuration for peer group report generation"""
    title: str = "InvestiGator Peer Group Analysis"
    subtitle: str = "AI-Powered Peer Comparison Investment Research"
    author: str = "InvestiGator AI System"
    include_charts: bool = True
    include_disclaimer: bool = True
    page_size: str = "letter"
    margin: float = 0.75 * inch


class PeerGroupPDFReportGenerator:
    """Generates comprehensive PDF reports for peer group analysis"""
    
    def __init__(self, output_dir: Path, config: Optional[PeerGroupReportConfig] = None):
        """
        Initialize peer group PDF report generator
        
        Args:
            output_dir: Directory for output reports
            config: Report configuration
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or PeerGroupReportConfig()
        
        # Initialize components
        self.chart_generator = ChartGenerator(
            self.output_dir.parent / "charts"
        )
        self.peer_analyzer = get_peer_comparison_analyzer()
        self.cache_manager = CacheManager()
        
        # Initialize styles
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles for peer group reports"""
        # Title style
        self.styles.add(ParagraphStyle(
            name='PeerGroupTitle',
            parent=self.styles['Title'],
            fontSize=24,
            textColor=colors.HexColor('#1a1a1a'),
            spaceAfter=12,
            alignment=TA_CENTER
        ))
        
        # Subtitle style
        self.styles.add(ParagraphStyle(
            name='PeerGroupSubtitle',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#444444'),
            spaceBefore=6,
            spaceAfter=12,
            alignment=TA_CENTER
        ))
        
        # Section header style
        self.styles.add(ParagraphStyle(
            name='PeerSectionHeader',
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
            name='PeerAnalysisText',
            parent=self.styles['BodyText'],
            fontSize=11,
            alignment=TA_JUSTIFY,
            spaceBefore=6,
            spaceAfter=6
        ))
        
        # Metric style
        self.styles.add(ParagraphStyle(
            name='PeerMetricText',
            parent=self.styles['BodyText'],
            fontSize=10,
            spaceBefore=4,
            spaceAfter=4
        ))

    def generate_peer_group_report(self, peer_group_data: Dict[str, Any], 
                                  symbol_recommendations: Dict[str, Any]) -> str:
        """
        Generate comprehensive PDF report for a peer group
        
        Args:
            peer_group_data: Peer group information and metrics
            symbol_recommendations: Individual symbol analysis results
            
        Returns:
            Path to generated PDF report
        """
        # Extract peer group info
        sector = peer_group_data['sector']
        industry = peer_group_data['industry']
        symbols = peer_group_data['symbols']
        
        # Generate peer group name for file (no timestamp)
        sector_clean = sector.replace('_', '').lower()
        industry_clean = industry.replace('_', '').lower()
        symbols_str = '-'.join(symbols[:3])
        others = '_others' if len(symbols) > 3 else ''
        
        filename = f"peer_group_{sector_clean}_{industry_clean}_{symbols_str}{others}.pdf"
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
        story.extend(self._create_peer_group_title_page(sector, industry, symbols))
        
        # Add executive summary
        story.extend(self._create_executive_summary(peer_group_data, symbol_recommendations))
        
        # Add peer group overview
        story.extend(self._create_peer_group_overview(peer_group_data))
        
        # Add peer comparison charts
        story.extend(self._create_peer_comparison_charts(symbols, symbol_recommendations))
        
        # Add peer relative valuation analysis (NEW)
        story.extend(self._create_peer_relative_analysis_section(peer_group_data, symbol_recommendations))
        
        # Add individual symbol analysis
        story.extend(self._create_individual_symbol_analysis(symbols, symbol_recommendations))
        
        # Add peer group metrics and commentary
        story.extend(self._create_peer_group_metrics_analysis(peer_group_data, symbol_recommendations))
        
        # Add disclaimers
        if self.config.include_disclaimer:
            story.extend(self._create_disclaimer())
        
        # Build PDF
        doc.build(story)
        
        logger.info(f"📄 Generated peer group PDF report: {filename}")
        return str(filepath)

    def _create_peer_group_title_page(self, sector: str, industry: str, symbols: List[str]) -> List:
        """Create title page for peer group report"""
        elements = []
        
        # Title
        title_text = f"{sector.title()} Sector Analysis"
        elements.append(Paragraph(title_text, self.styles['PeerGroupTitle']))
        elements.append(Spacer(1, 0.2 * inch))
        
        # Subtitle  
        subtitle_text = f"{industry.replace('_', ' ').title()} Peer Group"
        elements.append(Paragraph(subtitle_text, self.styles['PeerGroupSubtitle']))
        elements.append(Spacer(1, 0.3 * inch))
        
        # Symbol list
        symbols_text = f"<b>Companies Analyzed:</b> {', '.join(symbols)}"
        elements.append(Paragraph(symbols_text, self.styles['PeerAnalysisText']))
        elements.append(Spacer(1, 0.2 * inch))
        
        # Report info
        report_info = f"""
        <b>Report Generated:</b> {datetime.now().strftime('%B %d, %Y at %H:%M')}<br/>
        <b>Analysis Type:</b> Peer Group Comparative Analysis<br/>
        <b>Symbols Count:</b> {len(symbols)} companies<br/>
        <b>Generated by:</b> {self.config.author}
        """
        elements.append(Paragraph(report_info, self.styles['PeerAnalysisText']))
        
        elements.append(PageBreak())
        return elements

    def _create_executive_summary(self, peer_group_data: Dict[str, Any], 
                                symbol_recommendations: Dict[str, Any]) -> List:
        """Create executive summary section"""
        elements = []
        elements.append(Paragraph("Executive Summary", self.styles['PeerSectionHeader']))
        
        sector = peer_group_data['sector']
        industry = peer_group_data['industry']
        symbols = peer_group_data['symbols']
        
        # Calculate summary metrics
        scores = []
        recommendations = []
        for symbol in symbols:
            if symbol in symbol_recommendations:
                rec_data = symbol_recommendations[symbol]
                # Handle both old and new formats
                if isinstance(rec_data, dict):
                    if 'recommendation' in rec_data and rec_data['recommendation']:
                        # New format with full results
                        r = rec_data['recommendation']
                        if r:
                            scores.append(r.get('overall_score', 0))
                            recommendations.append(r.get('recommendation', 'HOLD'))
                    elif rec_data.get('status') == 'success' and rec_data.get('recommendation'):
                        # Old format
                        r = rec_data['recommendation']
                        scores.append(r.get('overall_score', 0))
                        recommendations.append(r.get('recommendation', 'HOLD'))
        
        avg_score = sum(scores) / len(scores) if scores else 0
        buy_count = sum(1 for r in recommendations if r == 'BUY')
        hold_count = sum(1 for r in recommendations if r == 'HOLD')
        sell_count = sum(1 for r in recommendations if r == 'SELL')
        
        # Sector overview
        sector_desc = self._get_sector_description(sector, industry)
        elements.append(Paragraph(f"<b>Sector Overview:</b> {sector_desc}", self.styles['PeerAnalysisText']))
        elements.append(Spacer(1, 0.1 * inch))
        
        # Summary metrics
        summary_text = f"""
        <b>Peer Group Performance Summary:</b><br/>
        • Average Investment Score: {avg_score:.1f}/10<br/>
        • Investment Recommendations: {buy_count} BUY, {hold_count} HOLD, {sell_count} SELL<br/>
        • Companies Analyzed: {len(symbols)} in {industry.replace('_', ' ').title()}<br/>
        • Sector: {sector.title()}
        """
        elements.append(Paragraph(summary_text, self.styles['PeerAnalysisText']))
        elements.append(Spacer(1, 0.2 * inch))
        
        # Key insights
        insights_text = self._generate_peer_group_insights(peer_group_data, symbol_recommendations)
        elements.append(Paragraph("<b>Key Insights:</b>", self.styles['PeerAnalysisText']))
        elements.append(Paragraph(insights_text, self.styles['PeerAnalysisText']))
        
        elements.append(Spacer(1, 0.3 * inch))
        return elements

    def _create_peer_group_overview(self, peer_group_data: Dict[str, Any]) -> List:
        """Create peer group overview section"""
        elements = []
        elements.append(Paragraph("Peer Group Overview", self.styles['PeerSectionHeader']))
        
        # Get peer group details from our data
        peer_info = self.peer_analyzer.get_peer_group(peer_group_data['symbols'][0])
        
        if peer_info and peer_info.get('peers'):
            # Full peer group composition
            large_cap = peer_info.get('large_cap', [])
            mid_cap = peer_info.get('mid_cap', [])
            
            overview_text = f"""
            <b>Industry Classification:</b> {peer_info['industry']}<br/>
            <b>Sector:</b> {peer_info['sector']}<br/>
            <b>Total Peer Universe:</b> {len(peer_info['peers'])} companies<br/>
            <b>Large Cap Companies:</b> {len(large_cap)} ({', '.join(large_cap[:5])}{'...' if len(large_cap) > 5 else ''})<br/>
            <b>Mid Cap Companies:</b> {len(mid_cap)} ({', '.join(mid_cap[:5])}{'...' if len(mid_cap) > 5 else ''})
            """
            elements.append(Paragraph(overview_text, self.styles['PeerAnalysisText']))
        
        elements.append(Spacer(1, 0.2 * inch))
        return elements

    def _create_peer_comparison_charts(self, symbols: List[str], 
                                     symbol_recommendations: Dict[str, Any]) -> List:
        """Create peer comparison charts section"""
        elements = []
        elements.append(Paragraph("Peer Group Positioning", self.styles['PeerSectionHeader']))
        
        # Prepare data for charts
        chart_data = []
        for symbol in symbols:
            if symbol in symbol_recommendations:
                rec_data = symbol_recommendations[symbol]
                r = None
                # Handle both old and new formats
                if isinstance(rec_data, dict):
                    if 'recommendation' in rec_data and rec_data['recommendation']:
                        # New format with full results
                        r = rec_data['recommendation']
                    elif rec_data.get('status') == 'success' and rec_data.get('recommendation'):
                        # Old format
                        r = rec_data['recommendation']
                
                if r:
                    chart_data.append({
                        'symbol': symbol,
                        'overall_score': r.get('overall_score', 0),
                        'fundamental_score': r.get('fundamental_score', 0),
                        'technical_score': r.get('technical_score', 0),
                        'income_score': r.get('income_score', 0),
                        'cashflow_score': r.get('cashflow_score', 0),
                        'balance_score': r.get('balance_score', 0)
                    })
        
        if chart_data:
            # Generate 3D fundamental plot for peer group
            fundamental_3d = self.chart_generator.generate_3d_fundamental_plot(chart_data)
            if fundamental_3d and os.path.exists(fundamental_3d):
                elements.append(Paragraph("<b>3D Fundamental Analysis Positioning</b>", self.styles['PeerAnalysisText']))
                elements.append(Spacer(1, 0.1 * inch))
                try:
                    img = Image(fundamental_3d, width=6*inch, height=4*inch)
                    elements.append(img)
                except:
                    logger.warning(f"Could not load 3D chart: {fundamental_3d}")
                elements.append(Spacer(1, 0.1 * inch))
                
                chart_desc = """
                The 3D chart above shows the relative positioning of peer group companies across three fundamental dimensions:
                Income Statement performance, Cash Flow strength, and Balance Sheet health. Companies positioned toward 
                the upper-right corner demonstrate superior fundamental metrics across all dimensions.
                """
                elements.append(Paragraph(chart_desc, self.styles['PeerMetricText']))
                elements.append(Spacer(1, 0.2 * inch))
            
            # Generate 2D technical vs fundamental plot
            tech_fund_2d = self.chart_generator.generate_2d_technical_fundamental_plot(chart_data)
            if tech_fund_2d and os.path.exists(tech_fund_2d):
                elements.append(Paragraph("<b>Technical vs Fundamental Positioning</b>", self.styles['PeerAnalysisText']))
                elements.append(Spacer(1, 0.1 * inch))
                try:
                    img = Image(tech_fund_2d, width=6*inch, height=4*inch)
                    elements.append(img)
                except:
                    logger.warning(f"Could not load 2D chart: {tech_fund_2d}")
                elements.append(Spacer(1, 0.1 * inch))
                
                chart_desc = """
                This 2D scatter plot compares Technical Analysis scores (price momentum, trends) against Fundamental Analysis 
                scores (financial health, valuation). The ideal investment zone is the upper-right quadrant, representing 
                companies with both strong fundamentals and positive technical momentum.
                """
                elements.append(Paragraph(chart_desc, self.styles['PeerMetricText']))
        
        elements.append(Spacer(1, 0.3 * inch))
        return elements

    def _create_individual_symbol_analysis(self, symbols: List[str], 
                                         symbol_recommendations: Dict[str, Any]) -> List:
        """Create individual symbol analysis section"""
        elements = []
        elements.append(Paragraph("Individual Company Analysis", self.styles['PeerSectionHeader']))
        
        # Create ranking table
        ranked_data = []
        for symbol in symbols:
            if symbol in symbol_recommendations:
                rec_data = symbol_recommendations[symbol]
                # Handle both old format (direct recommendation) and new format (full results)
                if isinstance(rec_data, dict):
                    if 'recommendation' in rec_data and rec_data['recommendation']:
                        # New format with full results
                        r = rec_data['recommendation']
                        if r:  # Ensure recommendation exists
                            ranked_data.append([
                                symbol,
                                f"{r.get('overall_score', 0):.1f}/10",
                                r.get('recommendation', 'HOLD'),
                                r.get('confidence', 'LOW'),
                                f"${r.get('price_target', 0):.2f}",
                                f"${r.get('current_price', 0):.2f}"
                            ])
                    elif rec_data.get('status') == 'success' and rec_data.get('recommendation'):
                        # Old format compatibility
                        r = rec_data['recommendation']
                        ranked_data.append([
                            symbol,
                            f"{r.get('overall_score', 0):.1f}/10",
                            r.get('recommendation', 'HOLD'),
                            r.get('confidence', 'LOW'),
                            f"${r.get('price_target', 0):.2f}",
                            f"${r.get('current_price', 0):.2f}"
                        ])
        
        # Sort by overall score
        ranked_data.sort(key=lambda x: float(x[1].split('/')[0]), reverse=True)
        
        # Add table headers
        table_data = [['Rank', 'Symbol', 'Score', 'Recommendation', 'Confidence', 'Target', 'Current']]
        for i, row in enumerate(ranked_data, 1):
            table_data.append([str(i)] + row)
        
        # Create table
        table = Table(table_data, colWidths=[0.6*inch, 0.8*inch, 1*inch, 1.2*inch, 1*inch, 1*inch, 1*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 9)
        ]))
        
        elements.append(table)
        elements.append(Spacer(1, 0.3 * inch))
        
        return elements

    def _create_peer_group_metrics_analysis(self, peer_group_data: Dict[str, Any], 
                                          symbol_recommendations: Dict[str, Any]) -> List:
        """Create peer group metrics and commentary section"""
        elements = []
        elements.append(Paragraph("Peer Group Metrics & Commentary", self.styles['PeerSectionHeader']))
        
        # Analyze peer group performance
        commentary = self._generate_peer_metrics_commentary(peer_group_data, symbol_recommendations)
        elements.append(Paragraph(commentary, self.styles['PeerAnalysisText']))
        
        elements.append(Spacer(1, 0.2 * inch))
        return elements

    def _create_disclaimer(self) -> List:
        """Create disclaimer section"""
        elements = []
        elements.append(PageBreak())
        elements.append(Paragraph("Important Disclaimers", self.styles['PeerSectionHeader']))
        
        disclaimer_text = """
        <b>Investment Risk Disclaimer:</b> This peer group analysis is for informational purposes only and does not 
        constitute investment advice, recommendation, or solicitation. All investments carry risk of loss. Past 
        performance does not guarantee future results. Peer group comparisons are based on current market data 
        and may change rapidly.
        
        <b>Data Sources:</b> Analysis is based on SEC filings, market data, and AI-powered analysis. While we strive 
        for accuracy, data may contain errors or omissions. Always verify information independently before making 
        investment decisions.
        
        <b>AI Analysis Limitations:</b> This report contains AI-generated analysis and insights. AI models may have 
        biases, limitations, or errors. Human oversight and additional research are recommended for investment decisions.
        """
        elements.append(Paragraph(disclaimer_text, self.styles['PeerMetricText']))
        
        return elements

    def _get_sector_description(self, sector: str, industry: str) -> str:
        """Get description for sector/industry"""
        descriptions = {
            'financials': {
                'banks_money_center': 'Large diversified banks providing comprehensive financial services including retail banking, commercial lending, investment banking, and wealth management.',
                'banks_regional': 'Regional and specialty banks focused on specific geographic markets or banking niches.',
                'insurance_property': 'Property and casualty insurance companies providing coverage for homes, automobiles, and commercial properties.',
                'insurance_life': 'Life insurance companies offering life insurance, annuities, and retirement planning products.',
                'asset_management': 'Investment management companies providing portfolio management, advisory services, and financial products.',
                'payment_processing': 'Companies operating payment networks, credit card processing, and digital payment solutions.'
            },
            'technology': {
                'software_infrastructure': 'Enterprise software companies providing cloud platforms, business applications, and IT infrastructure solutions.',
                'semiconductors': 'Companies designing and manufacturing computer chips, processors, and semiconductor components.',
                'consumer_electronics': 'Manufacturers of consumer electronic devices including smartphones, computers, and entertainment systems.'
            },
            'healthcare': {
                'pharmaceuticals': 'Large pharmaceutical companies developing, manufacturing, and marketing prescription drugs and medical treatments.',
                'biotech': 'Biotechnology companies focused on developing innovative therapies using biological processes.',
                'medical_devices': 'Companies manufacturing medical equipment, devices, and diagnostic tools for healthcare providers.'
            }
        }
        
        return descriptions.get(sector, {}).get(industry, f"{industry.replace('_', ' ').title()} companies in the {sector.title()} sector.")

    def _generate_peer_group_insights(self, peer_group_data: Dict[str, Any], 
                                    symbol_recommendations: Dict[str, Any]) -> str:
        """Generate key insights about the peer group"""
        symbols = peer_group_data['symbols']
        sector = peer_group_data['sector']
        
        # Analyze recommendations
        scores = []
        recommendations = []
        for symbol in symbols:
            if symbol in symbol_recommendations:
                rec_data = symbol_recommendations[symbol]
                r = None
                # Handle both old and new formats
                if isinstance(rec_data, dict):
                    if 'recommendation' in rec_data and rec_data['recommendation']:
                        # New format with full results
                        r = rec_data['recommendation']
                    elif rec_data.get('status') == 'success' and rec_data.get('recommendation'):
                        # Old format
                        r = rec_data['recommendation']
                
                if r:
                    scores.append(r.get('overall_score', 0))
                    recommendations.append(r.get('recommendation', 'HOLD'))
        
        if not scores:
            return "Insufficient data available for peer group analysis."
        
        avg_score = sum(scores) / len(scores)
        min_score = min(scores)
        max_score = max(scores)
        buy_count = sum(1 for r in recommendations if r == 'BUY')
        
        insights = []
        
        # Score distribution insight
        if max_score - min_score > 2.0:
            insights.append(f"Significant performance dispersion within the peer group (scores range {min_score:.1f} to {max_score:.1f})")
        else:
            insights.append(f"Relatively consistent performance across peer group (scores range {min_score:.1f} to {max_score:.1f})")
        
        # Recommendation distribution
        if buy_count == len(recommendations):
            insights.append("Strong sector outlook with all companies receiving BUY recommendations")
        elif buy_count > len(recommendations) * 0.7:
            insights.append("Generally positive sector sentiment with majority BUY recommendations")
        else:
            insights.append("Mixed sector outlook with varied investment recommendations")
        
        # Sector-specific insights
        if sector == 'financials':
            insights.append("Financial sector companies evaluated on regulatory environment, interest rate sensitivity, and credit quality")
        elif sector == 'technology':
            insights.append("Technology companies assessed on innovation capabilities, market position, and growth sustainability")
        elif sector == 'healthcare':
            insights.append("Healthcare companies analyzed for pipeline strength, regulatory approvals, and market expansion potential")
        
        return "• " + "<br/>• ".join(insights)

    def _extract_financial_metrics_from_cache(self, symbol: str) -> Dict[str, Any]:
        """Extract financial metrics from cached synthesis/fundamental data"""
        try:
            # Try to read synthesis response directly from file cache
            synthesis_file = Path(f"data/llm_cache/{symbol}/response_synthesis.txt")
            if synthesis_file.exists() and synthesis_file.stat().st_size > 0:
                with open(synthesis_file, 'r') as f:
                    synthesis_text = f.read()
                    metrics = self._parse_synthesis_metrics(synthesis_text, symbol)
                    if metrics and len(metrics) > 1:  # More than just symbol
                        return metrics
            
            # Try compressed synthesis file
            synthesis_gz_files = list(Path(f"data/llm_cache/{symbol}/").glob("llmresponse_*_full_*.json.gz"))
            if synthesis_gz_files:
                import gzip
                import json
                try:
                    with gzip.open(synthesis_gz_files[0], 'rt') as f:
                        cached_data = json.load(f)
                        if 'response' in cached_data and 'content' in cached_data['response']:
                            content = cached_data['response']['content']
                            # Convert the structured content to text for parsing
                            synthesis_text = f"""
                            Overall Score: {content.get('overall_score', 'N/A')}
                            Investment Thesis: {content.get('investment_thesis', '')}
                            Recommendation: {content.get('recommendation', 'N/A')}
                            Confidence: {content.get('confidence_level', 'N/A')}
                            """
                            metrics = self._parse_synthesis_metrics(synthesis_text, symbol)
                            if metrics and len(metrics) > 1:
                                # Add additional metrics from structured content
                                metrics['overall_score'] = content.get('overall_score')
                                metrics['recommendation'] = content.get('recommendation')
                                metrics['confidence'] = content.get('confidence_level')
                                return metrics
                except Exception as e:
                    logger.debug(f"Error parsing compressed synthesis for {symbol}: {e}")
            
            # Fallback: Try to extract from any available files with regex patterns
            return self._extract_metrics_from_any_cache_files(symbol)
                
        except Exception as e:
            logger.warning(f"Error extracting metrics for {symbol}: {e}")
            return {}

    def _extract_metrics_from_any_cache_files(self, symbol: str) -> Dict[str, Any]:
        """Extract metrics from any available cache files using regex patterns"""
        metrics = {"symbol": symbol}
        
        try:
            cache_dir = Path(f"data/llm_cache/{symbol}")
            if not cache_dir.exists():
                return metrics
            
            # Try all text files in the cache directory
            for cache_file in cache_dir.glob("*.txt"):
                if cache_file.stat().st_size > 0:
                    with open(cache_file, 'r') as f:
                        content = f.read()
                        extracted = self._parse_synthesis_metrics(content, symbol)
                        metrics.update(extracted)
            
            # Try compressed files
            for gz_file in cache_dir.glob("*.json.gz"):
                try:
                    import gzip
                    import json
                    with gzip.open(gz_file, 'rt') as f:
                        cached_data = json.load(f)
                        if 'response' in cached_data:
                            if isinstance(cached_data['response'], dict) and 'content' in cached_data['response']:
                                # Structured response
                                content = cached_data['response']['content']
                                if isinstance(content, dict):
                                    metrics['overall_score'] = content.get('overall_score')
                                    metrics['recommendation'] = content.get('recommendation')
                                    metrics['confidence'] = content.get('confidence_level')
                            elif isinstance(cached_data['response'], str):
                                # Text response
                                extracted = self._parse_synthesis_metrics(cached_data['response'], symbol)
                                metrics.update(extracted)
                except Exception as e:
                    logger.debug(f"Error parsing {gz_file}: {e}")
                    continue
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Error extracting from cache files for {symbol}: {e}")
            return metrics

    def _parse_synthesis_metrics(self, synthesis_data: str, symbol: str) -> Dict[str, Any]:
        """Parse financial metrics from synthesis response"""
        metrics = {"symbol": symbol}
        
        try:
            # Extract common financial ratios using more flexible regex patterns
            patterns = {
                "pe_ratio": [r"P/E.*?ratio.*?(\d+\.?\d*)", r"price.{0,5}earnings.*?(\d+\.?\d*)", r"P/E.*?(\d+\.?\d*)"],
                "pb_ratio": [r"P/B.*?ratio.*?(\d+\.?\d*)", r"price.{0,5}book.*?(\d+\.?\d*)", r"P/B.*?(\d+\.?\d*)"],
                "debt_to_equity": [r"debt.{0,15}equity.*?(\d+\.?\d*)", r"D/E.*?(\d+\.?\d*)", r"leverage.*?(\d+\.?\d*)"],
                "roe": [r"ROE.*?(\d+\.?\d*)%?", r"return.{0,5}equity.*?(\d+\.?\d*)%?"],
                "roa": [r"ROA.*?(\d+\.?\d*)%?", r"return.{0,5}assets.*?(\d+\.?\d*)%?"],
                "current_ratio": [r"current.{0,10}ratio.*?(\d+\.?\d*)", r"liquidity.*?ratio.*?(\d+\.?\d*)"],
                "revenue_growth": [r"revenue.{0,15}growth.*?(\d+\.?\d*)%?", r"sales.{0,15}growth.*?(\d+\.?\d*)%?"],
                "eps_growth": [r"EPS.{0,15}growth.*?(\d+\.?\d*)%?", r"earnings.{0,15}growth.*?(\d+\.?\d*)%?"],
                "gross_margin": [r"gross.{0,10}margin.*?(\d+\.?\d*)%?"],
                "net_margin": [r"net.{0,10}margin.*?(\d+\.?\d*)%?", r"profit.{0,10}margin.*?(\d+\.?\d*)%?"],
                "price_target": [r"target.*?price.*?\$?(\d+\.?\d*)", r"price.{0,10}target.*?\$?(\d+\.?\d*)"],
                "current_price": [r"current.*?price.*?\$?(\d+\.?\d*)", r"trading.*?at.*?\$?(\d+\.?\d*)"],
                "overall_score": [r"overall.{0,10}score.*?(\d+\.?\d*)", r"investment.{0,10}score.*?(\d+\.?\d*)"],
                "market_cap": [r"market.{0,10}cap.*?\$?(\d+\.?\d*)", r"valuation.*?\$?(\d+\.?\d*)"]
            }
            
            for metric, pattern_list in patterns.items():
                for pattern in pattern_list:
                    match = re.search(pattern, synthesis_data, re.IGNORECASE)
                    if match:
                        try:
                            value = float(match.group(1))
                            metrics[metric] = value
                            break  # Found a match, move to next metric
                        except (ValueError, IndexError):
                            continue
            
            # Also try to extract recommendation and confidence if not already present
            if 'recommendation' not in metrics:
                rec_match = re.search(r"recommendation.*?(?:BUY|SELL|HOLD)", synthesis_data, re.IGNORECASE)
                if rec_match:
                    metrics['recommendation'] = rec_match.group(0).split()[-1].upper()
            
            if 'confidence' not in metrics:
                conf_match = re.search(r"confidence.*?(HIGH|MEDIUM|LOW)", synthesis_data, re.IGNORECASE)
                if conf_match:
                    metrics['confidence'] = conf_match.group(1).upper()
                        
            return metrics
            
        except Exception as e:
            logger.warning(f"Error parsing synthesis metrics for {symbol}: {e}")
            return metrics

    def _parse_fundamental_metrics(self, fundamental_data: str, symbol: str) -> Dict[str, Any]:
        """Parse financial metrics from fundamental analysis response"""
        metrics = {"symbol": symbol}
        
        try:
            # Similar regex patterns for fundamental data
            patterns = {
                "total_revenue": r"revenue.*?\$?(\d+\.?\d*)\s*[MB]?",
                "net_income": r"net.{0,10}income.*?\$?(\d+\.?\d*)\s*[MB]?",
                "total_assets": r"total.{0,10}assets.*?\$?(\d+\.?\d*)\s*[MB]?",
                "total_debt": r"total.{0,10}debt.*?\$?(\d+\.?\d*)\s*[MB]?",
                "shareholders_equity": r"equity.*?\$?(\d+\.?\d*)\s*[MB]?",
                "eps": r"EPS.*?\$?(\d+\.?\d*)",
                "book_value": r"book.{0,10}value.*?\$?(\d+\.?\d*)"
            }
            
            for metric, pattern in patterns.items():
                match = re.search(pattern, fundamental_data, re.IGNORECASE)
                if match:
                    try:
                        value = float(match.group(1))
                        metrics[metric] = value
                    except (ValueError, IndexError):
                        continue
                        
            return metrics
            
        except Exception as e:
            logger.warning(f"Error parsing fundamental metrics for {symbol}: {e}")
            return metrics

    def _calculate_peer_relative_metrics(self, symbol_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate peer group averages and relative positioning"""
        if not symbol_metrics:
            return {}
            
        # Aggregate metrics by type
        metric_aggregates = {}
        
        # Financial ratios to calculate peer averages for
        ratio_metrics = [
            'pe_ratio', 'pb_ratio', 'debt_to_equity', 'roe', 'roa', 
            'current_ratio', 'revenue_growth', 'eps_growth', 
            'gross_margin', 'net_margin'
        ]
        
        for metric in ratio_metrics:
            values = []
            for symbol_data in symbol_metrics:
                if metric in symbol_data and symbol_data[metric] is not None:
                    values.append(symbol_data[metric])
            
            if values:
                metric_aggregates[metric] = {
                    'peer_average': statistics.mean(values),
                    'peer_median': statistics.median(values),
                    'peer_min': min(values),
                    'peer_max': max(values),
                    'peer_std': statistics.stdev(values) if len(values) > 1 else 0,
                    'peer_count': len(values)
                }
        
        return metric_aggregates

    def _calculate_relative_valuation(self, symbol: str, symbol_metrics: Dict[str, Any], 
                                    peer_averages: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate relative valuation and adjusted price targets"""
        relative_analysis = {
            'symbol': symbol,
            'discount_premium_analysis': {},
            'adjusted_price_target': None,
            'relative_score': 0
        }
        
        # Calculate discount/premium for key valuation metrics
        valuation_metrics = ['pe_ratio', 'pb_ratio', 'debt_to_equity']
        
        for metric in valuation_metrics:
            if metric in symbol_metrics and metric in peer_averages:
                symbol_value = symbol_metrics[metric]
                peer_avg = peer_averages[metric]['peer_average']
                
                if peer_avg > 0:
                    premium_discount = ((symbol_value - peer_avg) / peer_avg) * 100
                    
                    # For debt_to_equity, lower is better, so invert the logic
                    if metric == 'debt_to_equity':
                        premium_discount = -premium_discount
                    
                    relative_analysis['discount_premium_analysis'][metric] = {
                        'symbol_value': symbol_value,
                        'peer_average': peer_avg,
                        'premium_discount_pct': premium_discount,
                        'status': 'discount' if premium_discount < -5 else 'premium' if premium_discount > 5 else 'fair'
                    }
        
        # Calculate adjusted price target based on peer relative valuation
        if 'current_price' in symbol_metrics and 'price_target' in symbol_metrics:
            current_price = symbol_metrics['current_price']
            original_target = symbol_metrics['price_target']
            
            # Calculate average discount/premium
            premium_discounts = [
                analysis['premium_discount_pct'] 
                for analysis in relative_analysis['discount_premium_analysis'].values()
            ]
            
            if premium_discounts:
                avg_premium_discount = statistics.mean(premium_discounts)
                
                # Adjust price target based on peer relative positioning
                # If trading at discount to peers, potentially higher upside
                # If trading at premium, potentially lower upside
                peer_adjustment_factor = 1 + (avg_premium_discount / 100) * 0.5  # 50% weight to peer positioning
                
                adjusted_target = original_target * peer_adjustment_factor
                
                relative_analysis['adjusted_price_target'] = {
                    'original_target': original_target,
                    'adjusted_target': adjusted_target,
                    'adjustment_factor': peer_adjustment_factor,
                    'adjustment_reasoning': f"{'Discount' if avg_premium_discount < 0 else 'Premium'} to peer average of {abs(avg_premium_discount):.1f}%"
                }
        
        # Calculate overall relative score (0-10 scale)
        performance_metrics = ['roe', 'roa', 'revenue_growth', 'eps_growth', 'gross_margin', 'net_margin']
        relative_scores = []
        
        for metric in performance_metrics:
            if metric in symbol_metrics and metric in peer_averages:
                symbol_value = symbol_metrics[metric]
                peer_avg = peer_averages[metric]['peer_average']
                
                if peer_avg > 0:
                    relative_score = (symbol_value / peer_avg) * 5  # Scale to 0-10
                    relative_scores.append(min(10, max(0, relative_score)))
        
        if relative_scores:
            relative_analysis['relative_score'] = statistics.mean(relative_scores)
        
        return relative_analysis

    def _create_peer_relative_analysis_section(self, peer_group_data: Dict[str, Any], 
                                             symbol_recommendations: Dict[str, Any]) -> List:
        """Create comprehensive peer relative analysis section"""
        elements = []
        elements.append(Paragraph("Peer Relative Valuation Analysis", self.styles['PeerSectionHeader']))
        
        symbols = peer_group_data['symbols']
        
        # Extract financial metrics for all symbols
        all_symbol_metrics = []
        symbol_relative_analysis = {}
        
        for symbol in symbols:
            metrics = self._extract_financial_metrics_from_cache(symbol)
            if metrics:
                all_symbol_metrics.append(metrics)
        
        if len(all_symbol_metrics) < 2:
            elements.append(Paragraph(
                "Insufficient cached data for comprehensive peer relative analysis. "
                "Please run full analysis pipeline first.", 
                self.styles['PeerAnalysisText']
            ))
            return elements
        
        # Calculate peer averages
        peer_averages = self._calculate_peer_relative_metrics(all_symbol_metrics)
        
        # Calculate relative analysis for each symbol
        for symbol_data in all_symbol_metrics:
            symbol = symbol_data['symbol']
            relative_analysis = self._calculate_relative_valuation(symbol, symbol_data, peer_averages)
            symbol_relative_analysis[symbol] = relative_analysis
        
        # Create peer averages table
        elements.append(Paragraph("<b>Peer Group Averages:</b>", self.styles['PeerAnalysisText']))
        elements.append(Spacer(1, 0.1 * inch))
        
        avg_table_data = [['Metric', 'Peer Average', 'Range (Min-Max)', 'Std Dev']]
        
        for metric, data in peer_averages.items():
            if data['peer_count'] >= 2:
                metric_name = metric.replace('_', ' ').title()
                avg_val = f"{data['peer_average']:.2f}"
                range_val = f"{data['peer_min']:.2f} - {data['peer_max']:.2f}"
                std_val = f"{data['peer_std']:.2f}"
                avg_table_data.append([metric_name, avg_val, range_val, std_val])
        
        if len(avg_table_data) > 1:
            avg_table = Table(avg_table_data, colWidths=[2*inch, 1.2*inch, 1.5*inch, 1*inch])
            avg_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4472C4')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F2F2F2')),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            elements.append(avg_table)
            elements.append(Spacer(1, 0.2 * inch))
        
        # Create relative valuation table for each symbol
        elements.append(Paragraph("<b>Relative Valuation Analysis:</b>", self.styles['PeerAnalysisText']))
        elements.append(Spacer(1, 0.1 * inch))
        
        rel_table_data = [['Symbol', 'P/E vs Peers', 'P/B vs Peers', 'Debt/Equity vs Peers', 'Relative Score', 'Status']]
        
        for symbol, analysis in symbol_relative_analysis.items():
            discount_analysis = analysis['discount_premium_analysis']
            
            pe_status = "N/A"
            pb_status = "N/A"
            debt_status = "N/A"
            
            if 'pe_ratio' in discount_analysis:
                pe_pct = discount_analysis['pe_ratio']['premium_discount_pct']
                pe_status = f"{pe_pct:+.1f}%"
            
            if 'pb_ratio' in discount_analysis:
                pb_pct = discount_analysis['pb_ratio']['premium_discount_pct']
                pb_status = f"{pb_pct:+.1f}%"
                
            if 'debt_to_equity' in discount_analysis:
                debt_pct = discount_analysis['debt_to_equity']['premium_discount_pct']
                debt_status = f"{debt_pct:+.1f}%"
            
            rel_score = analysis['relative_score']
            status = "Strong" if rel_score > 7 else "Average" if rel_score > 4 else "Weak"
            
            rel_table_data.append([
                symbol, pe_status, pb_status, debt_status, 
                f"{rel_score:.1f}/10", status
            ])
        
        if len(rel_table_data) > 1:
            rel_table = Table(rel_table_data, colWidths=[0.8*inch, 1*inch, 1*inch, 1.2*inch, 1*inch, 0.8*inch])
            rel_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4472C4')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F2F2F2')),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            elements.append(rel_table)
            elements.append(Spacer(1, 0.2 * inch))
        
        # Add price target adjustments
        elements.append(Paragraph("<b>Peer-Adjusted Price Targets:</b>", self.styles['PeerAnalysisText']))
        elements.append(Spacer(1, 0.1 * inch))
        
        target_table_data = [['Symbol', 'Original Target', 'Adjusted Target', 'Adjustment', 'Reasoning']]
        
        for symbol, analysis in symbol_relative_analysis.items():
            if 'adjusted_price_target' in analysis and analysis['adjusted_price_target']:
                target_data = analysis['adjusted_price_target']
                target_table_data.append([
                    symbol,
                    f"${target_data['original_target']:.2f}",
                    f"${target_data['adjusted_target']:.2f}",
                    f"{((target_data['adjusted_target'] / target_data['original_target']) - 1) * 100:+.1f}%",
                    target_data['adjustment_reasoning']
                ])
        
        if len(target_table_data) > 1:
            target_table = Table(target_table_data, colWidths=[0.8*inch, 1.1*inch, 1.1*inch, 1*inch, 2.8*inch])
            target_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4472C4')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F2F2F2')),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTSIZE', (0, 1), (-1, -1), 8)
            ]))
            elements.append(target_table)
            elements.append(Spacer(1, 0.3 * inch))
        
        return elements

    def _generate_peer_metrics_commentary(self, peer_group_data: Dict[str, Any], 
                                        symbol_recommendations: Dict[str, Any]) -> str:
        """Generate detailed commentary on peer group metrics"""
        symbols = peer_group_data['symbols']
        sector = peer_group_data['sector']
        industry = peer_group_data['industry']
        
        commentary = f"""
        <b>Peer Group Analysis - {sector.title()} {industry.replace('_', ' ').title()}</b><br/><br/>
        
        <b>Comparative Performance:</b> This peer group analysis evaluates {len(symbols)} companies within the 
        {industry.replace('_', ' ').lower()} industry. Each company has been assessed using our comprehensive 
        investment framework incorporating fundamental analysis, technical indicators, and peer positioning.<br/><br/>
        
        <b>Methodology:</b> Our peer comparison leverages Russell 1000 industry classifications to ensure 
        relevant competitive comparisons. Companies are evaluated on financial health, growth prospects, 
        valuation metrics, and market positioning relative to industry peers.<br/><br/>
        
        <b>Key Metrics Focus:</b> For {sector.title()} sector companies, our analysis emphasizes:
        """
        
        # Add sector-specific metrics focus
        if sector == 'financials':
            commentary += """
            • Net Interest Margin and efficiency ratios<br/>
            • Loan loss provisions and credit quality<br/>
            • Return on Equity (ROE) and Return on Assets (ROA)<br/>
            • Regulatory capital adequacy<br/>
            • Fee income diversification
            """
        elif sector == 'technology':
            commentary += """
            • Revenue growth and recurring revenue quality<br/>
            • R&D investment and innovation pipeline<br/>
            • Market share and competitive positioning<br/>
            • Profitability margins and scalability<br/>
            • Technology platform moats
            """
        elif sector == 'healthcare':
            commentary += """
            • Drug pipeline strength and regulatory approvals<br/>
            • Patent protection and exclusivity periods<br/>
            • Market penetration and therapeutic expansion<br/>
            • R&D efficiency and success rates<br/>
            • Regulatory compliance and safety profiles
            """
        
        commentary += """<br/><br/>
        
        <b>Investment Implications:</b> Peer group analysis provides crucial context for individual investment 
        decisions by highlighting relative strengths, weaknesses, and positioning within the competitive landscape. 
        Companies demonstrating superior metrics relative to peers may indicate stronger investment potential, 
        while those lagging peer averages warrant additional scrutiny.<br/><br/>
        
        <b>Risk Considerations:</b> Peer group investments carry both systematic (sector-wide) and idiosyncratic 
        (company-specific) risks. Diversification across peer groups and sectors is recommended to manage 
        concentration risk and enhance portfolio resilience.
        """
        
        return commentary