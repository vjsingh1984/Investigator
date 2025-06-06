#!/usr/bin/env python3
"""
InvestiGator - SEC Data Adapter Patterns
Copyright (c) 2025 Vijaykumar Singh
Licensed under the Apache License 2.0

Adapter Pattern Implementations for SEC Data
Converts between different data formats (SEC API, Internal, LLM)
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from abc import ABC, abstractmethod

from data.models import QuarterlyData, FinancialStatementData, Filing
from utils.json_utils import safe_json_dumps
from config import get_config

logger = logging.getLogger(__name__)


class IDataFormatAdapter(ABC):
    """Interface for data format adapters"""
    
    @abstractmethod
    def adapt(self, data: Any) -> Any:
        """Adapt data from one format to another"""
        pass
    
    @abstractmethod
    def reverse_adapt(self, data: Any) -> Any:
        """Reverse adaptation (if supported)"""
        pass


class SECToInternalAdapter(IDataFormatAdapter):
    """Adapts SEC API format to internal data models"""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def adapt(self, sec_data: Dict[str, Any]) -> List[QuarterlyData]:
        """Convert SEC API response to internal QuarterlyData format"""
        quarterly_data = []
        
        # Handle different SEC response types
        if 'filings' in sec_data:
            quarterly_data = self._adapt_submissions(sec_data)
        elif 'facts' in sec_data:
            quarterly_data = self._adapt_company_facts(sec_data)
        
        return quarterly_data
    
    def _adapt_submissions(self, submissions_data: Dict) -> List[QuarterlyData]:
        """Adapt SEC submissions format"""
        quarterly_data = []
        
        recent_filings = submissions_data.get('filings', {}).get('recent', {})
        form_types = recent_filings.get('form', [])
        filing_dates = recent_filings.get('filingDate', [])
        accession_numbers = recent_filings.get('accessionNumber', [])
        
        cik = submissions_data.get('cik', '')
        entity_name = submissions_data.get('name', '')
        tickers = submissions_data.get('tickers', [])
        symbol = tickers[0] if tickers else ''
        
        for i, form_type in enumerate(form_types):
            if form_type in ['10-K', '10-Q']:
                # Parse fiscal period from filing metadata
                fiscal_year, fiscal_period = self._parse_fiscal_info(
                    form_type, 
                    filing_dates[i] if i < len(filing_dates) else None
                )
                
                qd = QuarterlyData(
                    symbol=symbol,
                    cik=str(cik).zfill(10),
                    fiscal_year=fiscal_year,
                    fiscal_period=fiscal_period,
                    form_type=form_type,
                    filing_date=filing_dates[i] if i < len(filing_dates) else '',
                    accession_number=accession_numbers[i] if i < len(accession_numbers) else '',
                    financial_data=FinancialStatementData(
                        symbol=symbol,
                        cik=str(cik).zfill(10),
                        fiscal_year=fiscal_year,
                        fiscal_period=fiscal_period
                    )
                )
                quarterly_data.append(qd)
        
        return quarterly_data
    
    def _adapt_company_facts(self, facts_data: Dict) -> List[QuarterlyData]:
        """Adapt SEC company facts format"""
        quarterly_data = []
        
        cik = str(facts_data.get('cik', '')).zfill(10)
        entity_name = facts_data.get('entityName', '')
        
        # Extract from us-gaap facts
        us_gaap = facts_data.get('facts', {}).get('us-gaap', {})
        
        # Use revenue concepts to identify periods
        revenue_concepts = ['Revenues', 'RevenueFromContractWithCustomerExcludingAssessedTax']
        
        periods_seen = set()
        
        for concept in revenue_concepts:
            if concept in us_gaap:
                units = us_gaap[concept].get('units', {})
                if 'USD' in units:
                    for entry in units['USD']:
                        form = entry.get('form', '')
                        if form in ['10-K', '10-Q']:
                            period_key = (entry.get('fy'), entry.get('fp'))
                            if period_key not in periods_seen:
                                periods_seen.add(period_key)
                                
                                qd = QuarterlyData(
                                    symbol='',  # Will need to be provided externally
                                    cik=cik,
                                    fiscal_year=entry.get('fy', 0),
                                    fiscal_period=entry.get('fp', ''),
                                    form_type=form,
                                    filing_date=entry.get('filed', ''),
                                    accession_number=entry.get('accn', ''),
                                    financial_data=FinancialStatementData(
                                        symbol='',
                                        cik=cik,
                                        fiscal_year=entry.get('fy', 0),
                                        fiscal_period=entry.get('fp', '')
                                    )
                                )
                                quarterly_data.append(qd)
                    break
        
        return quarterly_data
    
    def _parse_fiscal_info(self, form_type: str, filing_date: str) -> tuple:
        """Parse fiscal year and period from form type and filing date"""
        if not filing_date:
            return 0, ''
        
        try:
            year = int(filing_date[:4])
            
            if form_type == '10-K':
                return year, 'FY'
            else:
                # Estimate quarter based on filing month
                month = int(filing_date[5:7])
                if month <= 3:
                    return year - 1, 'Q4'
                elif month <= 6:
                    return year, 'Q1'
                elif month <= 9:
                    return year, 'Q2'
                else:
                    return year, 'Q3'
        except:
            return 0, ''
    
    def reverse_adapt(self, internal_data: List[QuarterlyData]) -> Dict[str, Any]:
        """Convert internal format back to SEC format (for compatibility)"""
        if not internal_data:
            return {}
        
        # Create SEC-like response
        forms = []
        filing_dates = []
        accession_numbers = []
        
        for qd in internal_data:
            forms.append(qd.form_type)
            filing_dates.append(qd.filing_date)
            accession_numbers.append(qd.accession_number)
        
        return {
            'filings': {
                'recent': {
                    'form': forms,
                    'filingDate': filing_dates,
                    'accessionNumber': accession_numbers
                }
            }
        }


class InternalToLLMAdapter(IDataFormatAdapter):
    """Adapts internal data format for LLM consumption"""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def adapt(self, quarterly_data: List[QuarterlyData]) -> str:
        """Convert quarterly data to LLM-friendly format"""
        if not quarterly_data:
            return "No financial data available"
        
        sections = []
        
        # Header
        symbol = quarterly_data[0].symbol
        sections.append(f"=== FINANCIAL DATA FOR {symbol} ===")
        sections.append(f"Periods Analyzed: {len(quarterly_data)}")
        sections.append("")
        
        # Process each quarter
        for qd in quarterly_data:
            sections.append(f"## {qd.fiscal_year} {qd.fiscal_period} (Filed: {qd.filing_date})")
            
            if qd.financial_data:
                # Income Statement
                if hasattr(qd.financial_data, 'income_statement') and qd.financial_data.income_statement:
                    sections.append("\n### Income Statement")
                    sections.extend(self._format_financial_section(qd.financial_data.income_statement))
                
                # Balance Sheet
                if hasattr(qd.financial_data, 'balance_sheet') and qd.financial_data.balance_sheet:
                    sections.append("\n### Balance Sheet")
                    sections.extend(self._format_financial_section(qd.financial_data.balance_sheet))
                
                # Cash Flow
                if hasattr(qd.financial_data, 'cash_flow_statement') and qd.financial_data.cash_flow_statement:
                    sections.append("\n### Cash Flow Statement")
                    sections.extend(self._format_financial_section(qd.financial_data.cash_flow_statement))
            
            sections.append("")
        
        return "\n".join(sections)
    
    def _format_financial_section(self, data: Dict[str, Any]) -> List[str]:
        """Format a financial statement section"""
        lines = []
        
        # Handle category-based structure (e.g., income_statement_primary, income_statement_secondary)
        for category_key, category_data in data.items():
            if isinstance(category_data, dict):
                # Check if this is a category with concepts and calculated_metrics
                if 'concepts' in category_data:
                    # Format category header
                    category_name = category_key.replace('_', ' ').title()
                    lines.append(f"\n#### {category_name}")
                    
                    # Format concepts
                    concepts = category_data.get('concepts', {})
                    for concept_key, concept_value in concepts.items():
                        if isinstance(concept_value, dict) and 'value' in concept_value:
                            amount = concept_value['value']
                            if isinstance(amount, (int, float)):
                                formatted_value = f"${amount:,.0f}"
                            else:
                                formatted_value = str(amount)
                            
                            display_key = concept_key.replace('_', ' ').title()
                            lines.append(f"- {display_key}: {formatted_value}")
                    
                    # Format calculated metrics if present
                    calculated_metrics = category_data.get('calculated_metrics', {})
                    if calculated_metrics:
                        lines.append("\n**Calculated Metrics:**")
                        for metric_key, metric_data in calculated_metrics.items():
                            if isinstance(metric_data, dict) and 'value' in metric_data:
                                value = metric_data['value']
                                if isinstance(value, (int, float)):
                                    # Special formatting for different metric types
                                    if 'eps' in metric_key.lower() or 'book_value_per_share' in metric_key.lower():
                                        formatted_value = f"${value:.2f}"
                                    elif 'margin' in metric_key.lower():
                                        formatted_value = f"{value:.1f}%"
                                    elif 'ratio' in metric_key.lower() or 'debt_to_equity' in metric_key.lower():
                                        formatted_value = f"{value:.2f}"
                                    elif 'working_capital' in metric_key.lower() or metric_data.get('unit') == 'USD':
                                        formatted_value = f"${value:,.0f}"
                                    else:
                                        formatted_value = f"{value:.2f}"
                                else:
                                    formatted_value = str(value)
                                
                                display_key = metric_key.replace('_', ' ').title()
                                
                                # Add calculation info if available
                                if 'calculation' in metric_data:
                                    lines.append(f"- {display_key}: {formatted_value} (calc: {metric_data['calculation']})")
                                else:
                                    lines.append(f"- {display_key}: {formatted_value}")
                
                # Handle old flat format for backward compatibility
                elif 'value' in category_data:
                    amount = category_data['value']
                    if isinstance(amount, (int, float)):
                        formatted_value = f"${amount:,.0f}"
                    else:
                        formatted_value = str(amount)
                    
                    display_key = category_key.replace('_', ' ').title()
                    lines.append(f"- {display_key}: {formatted_value}")
                    
            # Handle simple numeric values
            elif isinstance(category_data, (int, float)):
                formatted_value = f"${category_data:,.0f}"
                display_key = category_key.replace('_', ' ').title()
                lines.append(f"- {display_key}: {formatted_value}")
        
        return lines
    
    def reverse_adapt(self, llm_response: str) -> Dict[str, Any]:
        """Parse LLM response back to structured format"""
        # This would parse LLM output back to structured data
        # Implementation depends on LLM response format
        return {}


class FilingContentAdapter(IDataFormatAdapter):
    """Adapts raw filing content for different uses"""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def adapt(self, raw_html: str) -> str:
        """Clean and extract text from HTML filing"""
        try:
            # Use lxml if available for better parsing
            try:
                from lxml import html as lxml_html
                from lxml.html.clean import Cleaner
                
                cleaner = Cleaner(
                    scripts=True,
                    javascript=True,
                    comments=True,
                    style=True,
                    links=False,
                    meta=True,
                    page_structure=False,
                    processing_instructions=True,
                    embedded=True,
                    frames=True,
                    forms=True,
                    annoying_tags=True,
                    remove_unknown_tags=False,
                    safe_attrs_only=False
                )
                
                doc = lxml_html.fromstring(raw_html)
                cleaned = cleaner.clean_html(doc)
                text_content = cleaned.text_content()
                
            except ImportError:
                # Fallback to regex cleaning
                import re
                
                text_content = re.sub(r'<script[^>]*>.*?</script>', '', raw_html, flags=re.DOTALL | re.IGNORECASE)
                text_content = re.sub(r'<style[^>]*>.*?</style>', '', text_content, flags=re.DOTALL | re.IGNORECASE)
                text_content = re.sub(r'<[^>]+>', ' ', text_content)
                text_content = re.sub(r'\s+', ' ', text_content)
            
            # Clean up whitespace
            lines = [line.strip() for line in text_content.split('\n') if line.strip()]
            return '\n'.join(lines)
            
        except Exception as e:
            self.logger.error(f"Error cleaning filing content: {e}")
            return raw_html
    
    def reverse_adapt(self, text: str) -> str:
        """Convert text back to HTML (basic implementation)"""
        # Simple paragraph wrapping
        paragraphs = text.split('\n\n')
        html_parts = ['<html><body>']
        
        for para in paragraphs:
            if para.strip():
                html_parts.append(f'<p>{para.strip()}</p>')
        
        html_parts.append('</body></html>')
        return '\n'.join(html_parts)


class CompanyFactsToDetailedAdapter(IDataFormatAdapter):
    """Adapts company facts to detailed category format"""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.frame_details = getattr(self.config.sec, 'frame_api_details', {})
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def adapt(self, facts_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert company facts to detailed categories"""
        detailed_results = {}
        
        # Add company metadata
        detailed_results['company_metadata'] = {
            'entity_name': facts_data.get('entityName', ''),
            'cik': facts_data.get('cik', ''),
            'fiscal_year': 0,  # Will be set per period
            'fiscal_period': ''  # Will be set per period
        }
        
        # Process each category from frame_details
        for category_name, concept_mappings in self.frame_details.items():
            category_data = self._extract_category_data(facts_data, concept_mappings)
            if category_data:
                detailed_results[category_name] = category_data
        
        return detailed_results
    
    def _extract_category_data(self, facts: Dict, concept_mappings: Dict) -> Dict[str, Any]:
        """Extract data for a specific category"""
        concepts = {}
        successful = 0
        failed = 0
        
        us_gaap = facts.get('facts', {}).get('us-gaap', {})
        
        for field_name, concept_list in concept_mappings.items():
            value_found = False
            
            for concept in concept_list:
                clean_concept = concept.replace('us-gaap:', '')
                
                if clean_concept in us_gaap:
                    # Get most recent value
                    units = us_gaap[clean_concept].get('units', {})
                    
                    for unit_type in ['USD', 'shares', 'pure']:
                        if unit_type in units and units[unit_type]:
                            latest = max(units[unit_type], key=lambda x: x.get('end', ''))
                            
                            concepts[field_name] = {
                                'value': latest.get('val'),
                                'concept': clean_concept,
                                'unit': unit_type,
                                'form': latest.get('form', ''),
                                'filed': latest.get('filed', ''),
                                'accn': latest.get('accn', '')
                            }
                            successful += 1
                            value_found = True
                            break
                
                if value_found:
                    break
            
            if not value_found:
                concepts[field_name] = {
                    'value': '',
                    'concept': concept_list[0] if concept_list else '',
                    'unit': 'USD',
                    'missing': True
                }
                failed += 1
        
        return {
            'concepts': concepts,
            'metadata': {
                'successful': successful,
                'failed': failed,
                'total': successful + failed
            }
        }
    
    def reverse_adapt(self, detailed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert detailed format back to company facts format"""
        # This would reconstruct company facts structure from detailed categories
        # Implementation would be complex and rarely needed
        return {}