#!/usr/bin/env python3
"""
InvestiGator - Financial Data Aggregator Module
Copyright (c) 2025 Vijaykumar Singh
Licensed under the Apache License 2.0

Financial Data Aggregator Module
Handles aggregation and analysis of quarterly financial data for fundamental analysis
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json
from pathlib import Path

from config import get_config
from utils.sec_quarterly_processor import QuarterlyData

logger = logging.getLogger(__name__)

class FinancialDataAggregator:
    """
    Aggregates and analyzes quarterly financial data for fundamental analysis.
    
    This class handles:
    1. Consolidating quarterly data across periods
    2. Calculating financial ratios and metrics
    3. Trend analysis across quarters
    4. Data validation and quality checks
    """
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.main_logger = self.config.get_main_logger('financial_aggregator')
    
    def aggregate_quarterly_data(self, quarterly_data: List[QuarterlyData]) -> Dict[str, Any]:
        """
        Aggregate quarterly data into a comprehensive financial analysis.
        
        Args:
            quarterly_data: List of QuarterlyData objects
            
        Returns:
            Aggregated financial data dictionary
        """
        try:
            if not quarterly_data:
                return self._create_empty_aggregation()
            
            symbol = quarterly_data[0].symbol
            symbol_logger = self.config.get_symbol_logger(symbol, 'financial_aggregator')
            
            symbol_logger.info(f"Aggregating {len(quarterly_data)} quarters of data")
            
            aggregated = {
                'symbol': symbol,
                'analysis_date': datetime.utcnow().isoformat(),
                'quarters_analyzed': len(quarterly_data),
                'periods': [qd.get_period_key() for qd in quarterly_data],
                'income_statement': {},
                'balance_sheet': {},
                'cash_flow': {},
                'financial_ratios': {},
                'trends': {},
                'data_quality': {},
                'raw_quarters': [qd.to_dict() for qd in quarterly_data]
            }
            
            # Aggregate income statement data
            aggregated['income_statement'] = self._aggregate_income_statement(quarterly_data)
            
            # Aggregate balance sheet data
            aggregated['balance_sheet'] = self._aggregate_balance_sheet(quarterly_data)
            
            # Aggregate cash flow data
            aggregated['cash_flow'] = self._aggregate_cash_flow(quarterly_data)
            
            # Calculate financial ratios
            aggregated['financial_ratios'] = self._calculate_ratios(quarterly_data)
            
            # Analyze trends
            aggregated['trends'] = self._analyze_trends(quarterly_data)
            
            # Assess data quality
            aggregated['data_quality'] = self._assess_data_quality(quarterly_data)
            
            return aggregated
            
        except Exception as e:
            self.main_logger.error(f"Error aggregating quarterly data: {e}")
            return self._create_empty_aggregation()
    
    def _aggregate_income_statement(self, quarterly_data: List[QuarterlyData]) -> Dict[str, Any]:
        """Aggregate income statement data across quarters"""
        try:
            income_data = {
                'revenue': [],
                'cost_of_revenue': [],
                'gross_profit': [],
                'operating_expenses': [],
                'operating_income': [],
                'net_income': [],
                'earnings_per_share': [],
                'metrics': {}
            }
            
            for qd in quarterly_data:
                period_key = qd.get_period_key()
                
                # Extract revenue data
                revenue_data = qd.financial_data.get('income_statement_revenue', {})
                revenue_value = self._extract_concept_value(revenue_data, 'revenues')
                income_data['revenue'].append({
                    'period': period_key,
                    'value': revenue_value,
                    'fiscal_year': qd.fiscal_year,
                    'fiscal_period': qd.fiscal_period
                })
                
                # Extract cost of revenue
                cost_data = qd.financial_data.get('income_statement_cost_of_revenue', {})
                cost_value = self._extract_concept_value(cost_data, 'cost_of_revenue')
                income_data['cost_of_revenue'].append({
                    'period': period_key,
                    'value': cost_value
                })
                
                # Extract gross profit
                gross_data = qd.financial_data.get('income_statement_gross_profit', {})
                gross_value = self._extract_concept_value(gross_data, 'gross_profit')
                income_data['gross_profit'].append({
                    'period': period_key,
                    'value': gross_value
                })
                
                # Extract net income
                net_data = qd.financial_data.get('income_statement_net_income', {})
                net_value = self._extract_concept_value(net_data, 'net_income_loss')
                income_data['net_income'].append({
                    'period': period_key,
                    'value': net_value
                })
                
                # Extract EPS
                eps_value = self._extract_concept_value(net_data, 'earnings_per_share_diluted')
                income_data['earnings_per_share'].append({
                    'period': period_key,
                    'value': eps_value
                })
            
            # Calculate metrics
            income_data['metrics'] = self._calculate_income_metrics(income_data)
            
            return income_data
            
        except Exception as e:
            self.main_logger.error(f"Error aggregating income statement: {e}")
            return {}
    
    def _aggregate_balance_sheet(self, quarterly_data: List[QuarterlyData]) -> Dict[str, Any]:
        """Aggregate balance sheet data across quarters"""
        try:
            balance_data = {
                'total_assets': [],
                'current_assets': [],
                'total_liabilities': [],
                'current_liabilities': [],
                'shareholders_equity': [],
                'cash_and_equivalents': [],
                'metrics': {}
            }
            
            for qd in quarterly_data:
                period_key = qd.get_period_key()
                
                # Extract assets
                totals_data = qd.financial_data.get('balance_sheet_totals', {})
                assets_value = self._extract_concept_value(totals_data, 'total_assets')
                balance_data['total_assets'].append({
                    'period': period_key,
                    'value': assets_value
                })
                
                # Extract current assets
                current_assets_data = qd.financial_data.get('balance_sheet_current_assets', {})
                current_assets_value = self._extract_concept_value(current_assets_data, 'cash_and_cash_equivalents')
                balance_data['current_assets'].append({
                    'period': period_key,
                    'value': current_assets_value
                })
                
                # Extract liabilities
                liabilities_value = self._extract_concept_value(totals_data, 'total_liabilities')
                balance_data['total_liabilities'].append({
                    'period': period_key,
                    'value': liabilities_value
                })
                
                # Extract equity
                equity_value = self._extract_concept_value(totals_data, 'total_equity')
                balance_data['shareholders_equity'].append({
                    'period': period_key,
                    'value': equity_value
                })
            
            # Calculate metrics
            balance_data['metrics'] = self._calculate_balance_metrics(balance_data)
            
            return balance_data
            
        except Exception as e:
            self.main_logger.error(f"Error aggregating balance sheet: {e}")
            return {}
    
    def _aggregate_cash_flow(self, quarterly_data: List[QuarterlyData]) -> Dict[str, Any]:
        """Aggregate cash flow data across quarters"""
        try:
            cash_flow_data = {
                'operating_cash_flow': [],
                'investing_cash_flow': [],
                'financing_cash_flow': [],
                'capital_expenditures': [],
                'free_cash_flow': [],
                'metrics': {}
            }
            
            for qd in quarterly_data:
                period_key = qd.get_period_key()
                
                # Extract operating cash flow
                operating_data = qd.financial_data.get('cash_flow_operating', {})
                operating_value = self._extract_concept_value(operating_data, 'net_cash_provided_by_operating_activities')
                cash_flow_data['operating_cash_flow'].append({
                    'period': period_key,
                    'value': operating_value
                })
                
                # Extract investing cash flow
                investing_data = qd.financial_data.get('cash_flow_investing', {})
                investing_value = self._extract_concept_value(investing_data, 'net_cash_provided_by_investing_activities')
                cash_flow_data['investing_cash_flow'].append({
                    'period': period_key,
                    'value': investing_value
                })
                
                # Extract capital expenditures
                capex_value = self._extract_concept_value(investing_data, 'capital_expenditures')
                cash_flow_data['capital_expenditures'].append({
                    'period': period_key,
                    'value': capex_value
                })
                
                # Calculate free cash flow
                free_cf = None
                if operating_value and capex_value:
                    try:
                        free_cf = float(operating_value) - abs(float(capex_value))
                    except (ValueError, TypeError):
                        pass
                
                cash_flow_data['free_cash_flow'].append({
                    'period': period_key,
                    'value': free_cf
                })
            
            # Calculate metrics
            cash_flow_data['metrics'] = self._calculate_cash_flow_metrics(cash_flow_data)
            
            return cash_flow_data
            
        except Exception as e:
            self.main_logger.error(f"Error aggregating cash flow: {e}")
            return {}
    
    def _extract_concept_value(self, category_data: Dict, concept_name: str) -> Optional[float]:
        """Extract numeric value for a concept from category data"""
        try:
            concepts = category_data.get('concepts', {})
            concept_data = concepts.get(concept_name, {})
            
            value = concept_data.get('value')
            if value and not concept_data.get('missing', False):
                return float(value)
            
            return None
            
        except (ValueError, TypeError, KeyError):
            return None
    
    def _calculate_ratios(self, quarterly_data: List[QuarterlyData]) -> Dict[str, Any]:
        """Calculate financial ratios across quarters"""
        try:
            ratios = {
                'profitability': {},
                'liquidity': {},
                'efficiency': {},
                'leverage': {}
            }
            
            # Calculate ratios for each quarter
            for qd in quarterly_data:
                period_key = qd.get_period_key()
                
                # Get key values
                revenue = self._extract_concept_value(
                    qd.financial_data.get('income_statement_revenue', {}), 'revenues'
                )
                net_income = self._extract_concept_value(
                    qd.financial_data.get('income_statement_net_income', {}), 'net_income_loss'
                )
                total_assets = self._extract_concept_value(
                    qd.financial_data.get('balance_sheet_totals', {}), 'total_assets'
                )
                total_equity = self._extract_concept_value(
                    qd.financial_data.get('balance_sheet_totals', {}), 'total_equity'
                )
                
                # Calculate profitability ratios
                if revenue and net_income:
                    net_margin = (net_income / revenue) * 100
                    ratios['profitability'][period_key] = {
                        'net_profit_margin': net_margin
                    }
                
                # Calculate efficiency ratios
                if total_assets and net_income:
                    roa = (net_income / total_assets) * 100
                    if 'efficiency' not in ratios:
                        ratios['efficiency'] = {}
                    if period_key not in ratios['efficiency']:
                        ratios['efficiency'][period_key] = {}
                    ratios['efficiency'][period_key]['return_on_assets'] = roa
                
                # Calculate leverage ratios
                if total_equity and net_income:
                    roe = (net_income / total_equity) * 100
                    if 'leverage' not in ratios:
                        ratios['leverage'] = {}
                    if period_key not in ratios['leverage']:
                        ratios['leverage'][period_key] = {}
                    ratios['leverage'][period_key]['return_on_equity'] = roe
            
            return ratios
            
        except Exception as e:
            self.main_logger.error(f"Error calculating ratios: {e}")
            return {}
    
    def _analyze_trends(self, quarterly_data: List[QuarterlyData]) -> Dict[str, Any]:
        """Analyze trends across quarters"""
        try:
            trends = {
                'revenue_growth': [],
                'profit_growth': [],
                'margin_trends': [],
                'summary': {}
            }
            
            # Sort by period for trend analysis
            sorted_data = sorted(quarterly_data, key=lambda x: (x.fiscal_year, x.fiscal_period))
            
            # Calculate quarter-over-quarter growth
            for i in range(1, len(sorted_data)):
                current = sorted_data[i]
                previous = sorted_data[i-1]
                
                # Revenue growth
                current_revenue = self._extract_concept_value(
                    current.financial_data.get('income_statement_revenue', {}), 'revenues'
                )
                previous_revenue = self._extract_concept_value(
                    previous.financial_data.get('income_statement_revenue', {}), 'revenues'
                )
                
                if current_revenue and previous_revenue and previous_revenue != 0:
                    revenue_growth = ((current_revenue - previous_revenue) / previous_revenue) * 100
                    trends['revenue_growth'].append({
                        'period': current.get_period_key(),
                        'growth_rate': revenue_growth
                    })
            
            # Calculate trend summaries
            if trends['revenue_growth']:
                avg_revenue_growth = sum(t['growth_rate'] for t in trends['revenue_growth']) / len(trends['revenue_growth'])
                trends['summary']['average_revenue_growth'] = avg_revenue_growth
            
            return trends
            
        except Exception as e:
            self.main_logger.error(f"Error analyzing trends: {e}")
            return {}
    
    def _assess_data_quality(self, quarterly_data: List[QuarterlyData]) -> Dict[str, Any]:
        """Assess the quality and completeness of financial data"""
        try:
            quality = {
                'completeness_score': 0,
                'missing_concepts': [],
                'data_issues': [],
                'recommendations': []
            }
            
            total_concepts = 0
            missing_concepts = 0
            
            for qd in quarterly_data:
                for category, data in qd.financial_data.items():
                    concepts = data.get('concepts', {})
                    for concept_name, concept_data in concepts.items():
                        total_concepts += 1
                        if concept_data.get('missing', False):
                            missing_concepts += 1
                            quality['missing_concepts'].append({
                                'period': qd.get_period_key(),
                                'category': category,
                                'concept': concept_name
                            })
            
            # Calculate completeness score
            if total_concepts > 0:
                quality['completeness_score'] = ((total_concepts - missing_concepts) / total_concepts) * 100
            
            # Add recommendations based on data quality
            if quality['completeness_score'] < 70:
                quality['recommendations'].append("Data completeness is low - consider additional data sources")
            
            if missing_concepts > total_concepts * 0.5:
                quality['recommendations'].append("High number of missing concepts - verify ticker and CIK mapping")
            
            return quality
            
        except Exception as e:
            self.main_logger.error(f"Error assessing data quality: {e}")
            return {'completeness_score': 0, 'error': str(e)}
    
    def _calculate_income_metrics(self, income_data: Dict) -> Dict[str, Any]:
        """Calculate income statement metrics"""
        metrics = {}
        
        try:
            # Calculate revenue growth
            revenue_values = [r['value'] for r in income_data['revenue'] if r['value'] is not None]
            if len(revenue_values) >= 2:
                recent = revenue_values[0]  # Most recent
                previous = revenue_values[1]  # Previous quarter
                if previous and previous != 0:
                    growth = ((recent - previous) / previous) * 100
                    metrics['revenue_growth_qoq'] = growth
            
            # Calculate average margins
            margins = []
            for i, revenue in enumerate(income_data['revenue']):
                if revenue['value'] and i < len(income_data['net_income']):
                    net_income = income_data['net_income'][i]['value']
                    if net_income:
                        margin = (net_income / revenue['value']) * 100
                        margins.append(margin)
            
            if margins:
                metrics['average_net_margin'] = sum(margins) / len(margins)
            
        except Exception as e:
            self.main_logger.error(f"Error calculating income metrics: {e}")
        
        return metrics
    
    def _calculate_balance_metrics(self, balance_data: Dict) -> Dict[str, Any]:
        """Calculate balance sheet metrics"""
        metrics = {}
        
        try:
            # Calculate asset growth
            asset_values = [a['value'] for a in balance_data['total_assets'] if a['value'] is not None]
            if len(asset_values) >= 2:
                recent = asset_values[0]
                previous = asset_values[1]
                if previous and previous != 0:
                    growth = ((recent - previous) / previous) * 100
                    metrics['asset_growth_qoq'] = growth
        
        except Exception as e:
            self.main_logger.error(f"Error calculating balance metrics: {e}")
        
        return metrics
    
    def _calculate_cash_flow_metrics(self, cash_flow_data: Dict) -> Dict[str, Any]:
        """Calculate cash flow metrics"""
        metrics = {}
        
        try:
            # Calculate free cash flow margin
            fcf_values = [f['value'] for f in cash_flow_data['free_cash_flow'] if f['value'] is not None]
            if fcf_values:
                metrics['average_free_cash_flow'] = sum(fcf_values) / len(fcf_values)
        
        except Exception as e:
            self.main_logger.error(f"Error calculating cash flow metrics: {e}")
        
        return metrics
    
    def _create_empty_aggregation(self) -> Dict[str, Any]:
        """Create empty aggregation structure"""
        return {
            'symbol': '',
            'analysis_date': datetime.utcnow().isoformat(),
            'quarters_analyzed': 0,
            'periods': [],
            'income_statement': {},
            'balance_sheet': {},
            'cash_flow': {},
            'financial_ratios': {},
            'trends': {},
            'data_quality': {'completeness_score': 0},
            'raw_quarters': []
        }

def get_financial_aggregator() -> FinancialDataAggregator:
    """Get financial data aggregator instance"""
    return FinancialDataAggregator()