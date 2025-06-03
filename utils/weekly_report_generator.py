#!/usr/bin/env python3
"""
InvestiGator - Weekly Report Generation Module
Copyright (c) 2025 Vijaykumar Singh
Licensed under the Apache License 2.0

Weekly Report Generation Module for InvestiGator
Handles creation of weekly portfolio summary reports
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import json

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, 
    PageBreak, Image, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_JUSTIFY

from .report_generator import PDFReportGenerator, ReportConfig, NumberedCanvas

logger = logging.getLogger(__name__)


class WeeklyReportGenerator(PDFReportGenerator):
    """Generates weekly portfolio summary reports"""
    
    def __init__(self, output_dir: Path, config: Optional[ReportConfig] = None):
        """Initialize weekly report generator"""
        # Update config for weekly reports
        if config is None:
            config = ReportConfig(
                title="InvestiGator Weekly Portfolio Report",
                subtitle="AI-Powered Weekly Investment Summary"
            )
        super().__init__(output_dir, config)
    
    def generate_weekly_report(self, portfolio_data: List[Dict], 
                             market_summary: Optional[Dict] = None,
                             performance_data: Optional[Dict] = None) -> str:
        """
        Generate weekly portfolio report
        
        Args:
            portfolio_data: List of stock analysis data
            market_summary: Optional market overview data
            performance_data: Optional portfolio performance metrics
            
        Returns:
            Path to generated PDF report
        """
        # Create filename with week ending date
        week_end = datetime.now()
        week_start = week_end - timedelta(days=7)
        filename = f"weekly_report_{week_end.strftime('%Y%m%d')}.pdf"
        filepath = self.output_dir / filename
        
        # Create document
        doc = SimpleDocTemplate(
            str(filepath),
            pagesize=letter,
            rightMargin=self.config.margin,
            leftMargin=self.config.margin,
            topMargin=self.config.margin,
            bottomMargin=self.config.margin
        )
        
        # Build content
        story = []
        
        # Title page
        story.extend(self._create_weekly_title_page(week_start, week_end))
        
        # Market overview
        if market_summary:
            story.append(PageBreak())
            story.extend(self._create_market_overview(market_summary))
        
        # Portfolio performance summary
        if performance_data:
            story.extend(self._create_performance_summary(performance_data))
        
        # Top movers section
        story.append(PageBreak())
        story.extend(self._create_top_movers(portfolio_data))
        
        # Sector analysis
        story.extend(self._create_sector_analysis(portfolio_data))
        
        # Individual stock summaries
        story.append(PageBreak())
        story.extend(self._create_stock_summaries(portfolio_data))
        
        # Recommendations summary
        story.append(PageBreak())
        story.extend(self._create_recommendations_summary(portfolio_data))
        
        # Risk assessment
        story.extend(self._create_risk_assessment(portfolio_data))
        
        # Upcoming events
        story.append(PageBreak())
        story.extend(self._create_upcoming_events(portfolio_data))
        
        # Build PDF
        doc.build(story, canvasmaker=NumberedCanvas)
        
        logger.info(f"📊 Generated weekly report: {filepath}")
        return str(filepath)
    
    def _create_weekly_title_page(self, week_start: datetime, week_end: datetime) -> List:
        """Create weekly report title page"""
        elements = []
        
        # Add title
        elements.append(Spacer(1, 2 * inch))
        elements.append(Paragraph(self.config.title, self.styles['CustomTitle']))
        elements.append(Paragraph(self.config.subtitle, self.styles['CustomSubtitle']))
        
        # Add week period
        elements.append(Spacer(1, 0.5 * inch))
        period_text = f"Week of {week_start.strftime('%B %d')} - {week_end.strftime('%B %d, %Y')}"
        elements.append(Paragraph(period_text, self.styles['Heading2']))
        
        # Add generation time
        elements.append(Spacer(1, 0.5 * inch))
        gen_time = datetime.now().strftime('%B %d, %Y at %I:%M %p')
        elements.append(Paragraph(f"Generated: {gen_time}", self.styles['Normal']))
        
        # Add summary stats preview
        elements.append(Spacer(1, 1 * inch))
        elements.append(Paragraph("Weekly Highlights", self.styles['Heading3']))
        
        return elements
    
    def _create_market_overview(self, market_summary: Dict) -> List:
        """Create market overview section"""
        elements = []
        
        elements.append(Paragraph("Market Overview", self.styles['SectionHeader']))
        elements.append(Spacer(1, 0.2 * inch))
        
        # Market indices table
        indices_data = [
            ['Index', 'Current', 'Week Change', 'YTD Change'],
            ['S&P 500', market_summary.get('sp500', 'N/A'), 
             market_summary.get('sp500_week_change', 'N/A'),
             market_summary.get('sp500_ytd_change', 'N/A')],
            ['NASDAQ', market_summary.get('nasdaq', 'N/A'),
             market_summary.get('nasdaq_week_change', 'N/A'),
             market_summary.get('nasdaq_ytd_change', 'N/A')],
            ['DOW', market_summary.get('dow', 'N/A'),
             market_summary.get('dow_week_change', 'N/A'),
             market_summary.get('dow_ytd_change', 'N/A')]
        ]
        
        indices_table = Table(indices_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
        indices_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        elements.append(indices_table)
        
        # Market commentary
        if market_summary.get('commentary'):
            elements.append(Spacer(1, 0.2 * inch))
            elements.append(Paragraph("<b>Market Commentary</b>", self.styles['Heading3']))
            elements.append(Paragraph(market_summary['commentary'], self.styles['AnalysisText']))
        
        return elements
    
    def _create_performance_summary(self, performance_data: Dict) -> List:
        """Create portfolio performance summary"""
        elements = []
        
        elements.append(Spacer(1, 0.3 * inch))
        elements.append(Paragraph("Portfolio Performance", self.styles['SectionHeader']))
        elements.append(Spacer(1, 0.2 * inch))
        
        # Performance metrics
        perf_text = f"""
        <b>Week Performance:</b> {performance_data.get('week_return', 'N/A')}<br/>
        <b>Month Performance:</b> {performance_data.get('month_return', 'N/A')}<br/>
        <b>YTD Performance:</b> {performance_data.get('ytd_return', 'N/A')}<br/>
        <b>Win Rate:</b> {performance_data.get('win_rate', 'N/A')}<br/>
        <b>Best Performer:</b> {performance_data.get('best_performer', 'N/A')}<br/>
        <b>Worst Performer:</b> {performance_data.get('worst_performer', 'N/A')}<br/>
        """
        
        elements.append(Paragraph(perf_text, self.styles['AnalysisText']))
        
        return elements
    
    def _create_top_movers(self, portfolio_data: List[Dict]) -> List:
        """Create top movers section"""
        elements = []
        
        elements.append(Paragraph("Top Movers This Week", self.styles['SectionHeader']))
        elements.append(Spacer(1, 0.2 * inch))
        
        # Calculate weekly changes
        movers = []
        for stock in portfolio_data:
            symbol = stock.get('symbol', 'N/A')
            week_change = stock.get('price_change_1w', 0)
            # Get current price from executive_summary if not at top level
            current_price = stock.get('current_price', 0)
            if current_price == 0:
                current_price = stock.get('executive_summary', {}).get('current_price', 0)
            # Get overall score from composite_scores
            overall_score = stock.get('overall_score', 0)
            if overall_score == 0:
                overall_score = stock.get('composite_scores', {}).get('overall_score', 0)
            
            movers.append({
                'symbol': symbol,
                'change': week_change,
                'price': current_price,
                'score': overall_score
            })
        
        # Sort by absolute change
        movers.sort(key=lambda x: abs(x['change']), reverse=True)
        
        # Top gainers
        elements.append(Paragraph("<b>Top Gainers</b>", self.styles['Heading3']))
        gainers = [m for m in movers if m['change'] > 0][:5]
        
        if gainers:
            gainers_data = [['Symbol', 'Price', 'Week Change', 'Score']]
            for mover in gainers:
                gainers_data.append([
                    mover['symbol'],
                    f"${mover['price']:.2f}" if mover['price'] is not None else "$0.00",
                    f"+{mover['change']:.2f}%" if mover['change'] is not None else "+0.00%",
                    f"{mover['score']:.1f}" if mover['score'] is not None else "0.0"
                ])
            
            gainers_table = Table(gainers_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
            self._apply_table_style(gainers_table, color_positive=True)
            elements.append(gainers_table)
        
        # Top losers
        elements.append(Spacer(1, 0.2 * inch))
        elements.append(Paragraph("<b>Top Losers</b>", self.styles['Heading3']))
        losers = [m for m in movers if m['change'] < 0][:5]
        
        if losers:
            losers_data = [['Symbol', 'Price', 'Week Change', 'Score']]
            for mover in losers:
                losers_data.append([
                    mover['symbol'],
                    f"${mover['price']:.2f}" if mover['price'] is not None else "$0.00",
                    f"{mover['change']:.2f}%" if mover['change'] is not None else "0.00%",
                    f"{mover['score']:.1f}" if mover['score'] is not None else "0.0"
                ])
            
            losers_table = Table(losers_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
            self._apply_table_style(losers_table, color_negative=True)
            elements.append(losers_table)
        
        return elements
    
    def _create_sector_analysis(self, portfolio_data: List[Dict]) -> List:
        """Create sector analysis section"""
        elements = []
        
        elements.append(Spacer(1, 0.3 * inch))
        elements.append(Paragraph("Sector Analysis", self.styles['SectionHeader']))
        elements.append(Spacer(1, 0.2 * inch))
        
        # Group by sector
        sector_groups = defaultdict(list)
        for stock in portfolio_data:
            sector = stock.get('sector', 'Unknown')
            if sector == 'Unknown':
                sector = stock.get('executive_summary', {}).get('sector', 'Unknown')
            sector_groups[sector].append(stock)
        
        # Calculate sector metrics
        sector_data = [['Sector', 'Holdings', 'Avg Score', 'Avg Week Change']]
        
        for sector, stocks in sector_groups.items():
            # Calculate average score safely
            scores = []
            for s in stocks:
                score = s.get('overall_score', 0)
                if score == 0:
                    score = s.get('composite_scores', {}).get('overall_score', 0)
                if score is not None:
                    scores.append(score)
            avg_score = sum(scores) / len(scores) if scores else 0
            
            # Calculate average change safely  
            changes = []
            for s in stocks:
                change = s.get('price_change_1w', 0)
                if change is not None:
                    changes.append(change)
            avg_change = sum(changes) / len(changes) if changes else 0
            
            sector_data.append([
                sector,
                str(len(stocks)),
                f"{avg_score:.1f}" if avg_score is not None else "0.0",
                f"{avg_change:+.2f}%" if avg_change is not None else "+0.00%"
            ])
        
        sector_table = Table(sector_data, colWidths=[2.5*inch, 1.2*inch, 1.2*inch, 1.5*inch])
        self._apply_table_style(sector_table)
        elements.append(sector_table)
        
        return elements
    
    def _create_stock_summaries(self, portfolio_data: List[Dict]) -> List:
        """Create individual stock summaries"""
        elements = []
        
        elements.append(Paragraph("Individual Stock Summaries", self.styles['SectionHeader']))
        elements.append(Spacer(1, 0.2 * inch))
        
        # Sort by overall score
        def get_score(stock):
            score = stock.get('overall_score', 0)
            if score == 0:
                score = stock.get('composite_scores', {}).get('overall_score', 0)
            return score if score is not None else 0
        
        sorted_stocks = sorted(portfolio_data, key=get_score, reverse=True)
        
        for stock in sorted_stocks[:10]:  # Top 10 stocks
            symbol = stock.get('symbol', 'N/A')
            score = get_score(stock)
            recommendation = stock.get('investment_recommendation', {}).get('recommendation', 'N/A')
            if recommendation == 'N/A':
                recommendation = stock.get('recommendation', 'N/A')
            
            # Get current price safely
            price = stock.get('current_price', 0)
            if price == 0:
                price = stock.get('executive_summary', {}).get('current_price', 0)
            
            # Get target price safely
            target = stock.get('price_target', 0)
            if target == 0:
                target = stock.get('investment_recommendation', {}).get('target_price', {}).get('12_month_target', 0)
            
            week_change = stock.get('price_change_1w', 0)
            
            # Stock header with safe formatting
            price_str = f"${price:.2f}" if price is not None else "$0.00"
            target_str = f"${target:.2f}" if target is not None else "$0.00"
            score_str = f"{score:.1f}" if score is not None else "0.0"
            change_str = f"{week_change:+.2f}%" if week_change is not None else "+0.00%"
            
            stock_text = f"""
            <b>{symbol}</b> - {recommendation} (Score: {score_str}/10)<br/>
            Current Price: {price_str} | Target: {target_str} | Week Change: {change_str}
            """
            elements.append(Paragraph(stock_text, self.styles['AnalysisText']))
            
            # Key insights (first 2)
            insights = stock.get('key_insights', [])[:2]
            if insights:
                for insight in insights:
                    elements.append(Paragraph(f"• {insight}", self.styles['Normal']))
            
            elements.append(Spacer(1, 0.1 * inch))
        
        return elements
    
    def _create_recommendations_summary(self, portfolio_data: List[Dict]) -> List:
        """Create recommendations summary"""
        elements = []
        
        elements.append(Paragraph("Action Items", self.styles['SectionHeader']))
        elements.append(Spacer(1, 0.2 * inch))
        
        # Group by recommendation
        buy_stocks = []
        sell_stocks = []
        
        for stock in portfolio_data:
            recommendation = stock.get('investment_recommendation', {}).get('recommendation', 'N/A')
            if recommendation == 'N/A':
                recommendation = stock.get('recommendation', 'N/A')
            
            if 'BUY' in recommendation.upper():
                buy_stocks.append(stock)
            elif 'SELL' in recommendation.upper():
                sell_stocks.append(stock)
        
        # Buy recommendations
        if buy_stocks:
            elements.append(Paragraph("<b>Buy Recommendations</b>", self.styles['Heading3']))
            for stock in buy_stocks[:5]:
                # Get current price safely
                current_price = stock.get('current_price', 0)
                if current_price == 0:
                    current_price = stock.get('executive_summary', {}).get('current_price', 0)
                
                # Get target price safely
                target_price = stock.get('price_target', 0)
                if target_price == 0:
                    target_price = stock.get('investment_recommendation', {}).get('target_price', {}).get('12_month_target', 0)
                
                current_str = f"${current_price:.2f}" if current_price is not None else "$0.00"
                target_str = f"${target_price:.2f}" if target_price is not None else "$0.00"
                
                entry_text = f"• <b>{stock['symbol']}</b>: Entry at {current_str}, Target {target_str}"
                elements.append(Paragraph(entry_text, self.styles['Normal']))
        
        # Sell recommendations
        if sell_stocks:
            elements.append(Spacer(1, 0.1 * inch))
            elements.append(Paragraph("<b>Sell Recommendations</b>", self.styles['Heading3']))
            for stock in sell_stocks[:5]:
                # Get current price safely
                current_price = stock.get('current_price', 0)
                if current_price == 0:
                    current_price = stock.get('executive_summary', {}).get('current_price', 0)
                
                # Get stop loss safely
                stop_loss = stock.get('stop_loss', 0)
                if stop_loss == 0:
                    stop_loss = stock.get('technical_assessment', {}).get('risk_management', {}).get('stop_loss_level', 0)
                
                current_str = f"${current_price:.2f}" if current_price is not None else "$0.00"
                stop_str = f"${stop_loss:.2f}" if stop_loss is not None else "$0.00"
                
                exit_text = f"• <b>{stock['symbol']}</b>: Exit at {current_str}, Stop at {stop_str}"
                elements.append(Paragraph(exit_text, self.styles['Normal']))
        
        return elements
    
    def _create_risk_assessment(self, portfolio_data: List[Dict]) -> List:
        """Create portfolio risk assessment"""
        elements = []
        
        elements.append(Spacer(1, 0.3 * inch))
        elements.append(Paragraph("Risk Assessment", self.styles['SectionHeader']))
        elements.append(Spacer(1, 0.2 * inch))
        
        # Aggregate risks
        all_risks = []
        for stock in portfolio_data:
            risks = stock.get('key_risks', [])
            for risk in risks:
                all_risks.append(f"{stock['symbol']}: {risk}")
        
        # Show top 5 risks
        risk_text = "Key portfolio risks to monitor:<br/>"
        for risk in all_risks[:5]:
            risk_text += f"• {risk}<br/>"
        
        elements.append(Paragraph(risk_text, self.styles['AnalysisText']))
        
        return elements
    
    def _create_upcoming_events(self, portfolio_data: List[Dict]) -> List:
        """Create upcoming events section"""
        elements = []
        
        elements.append(Paragraph("Upcoming Events", self.styles['SectionHeader']))
        elements.append(Spacer(1, 0.2 * inch))
        
        # Extract upcoming events (earnings, ex-dividend dates, etc.)
        events_text = """
        <b>Next Week Watch List:</b><br/>
        • Earnings releases scheduled<br/>
        • Key economic data releases<br/>
        • Technical levels to monitor<br/>
        • Sector rotation opportunities<br/>
        """
        
        elements.append(Paragraph(events_text, self.styles['AnalysisText']))
        
        return elements
    
    def _apply_table_style(self, table: Table, color_positive: bool = False, color_negative: bool = False):
        """Apply consistent table styling"""
        base_style = [
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 9)
        ]
        
        if color_positive:
            base_style.append(('BACKGROUND', (0, 1), (-1, -1), colors.lightgreen))
        elif color_negative:
            base_style.append(('BACKGROUND', (0, 1), (-1, -1), colors.lightpink))
        else:
            base_style.append(('BACKGROUND', (0, 1), (-1, -1), colors.beige))
        
        table.setStyle(TableStyle(base_style))