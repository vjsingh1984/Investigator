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
from data.models import QuarterlyData

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
                'periods': [qd.period_key for qd in quarterly_data],
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
                period_key = qd.period_key
                
                # Get income statement data directly from the new structure
                if hasattr(qd.financial_data, 'income_statement') and qd.financial_data.income_statement:
                    income_statement = qd.financial_data.income_statement
                    
                    # Extract revenue
                    revenue_value = income_statement.get('revenue', 0)
                    income_data['revenue'].append({
                        'period': period_key,
                        'value': revenue_value,
                        'fiscal_year': qd.fiscal_year,
                        'fiscal_period': qd.fiscal_period
                    })
                    
                    # Extract cost of revenue
                    cost_value = income_statement.get('cost_of_revenue', 0)
                    income_data['cost_of_revenue'].append({
                        'period': period_key,
                        'value': cost_value
                    })
                    
                    # Extract gross profit
                    gross_value = income_statement.get('gross_profit', 0)
                    income_data['gross_profit'].append({
                        'period': period_key,
                        'value': gross_value
                    })
                    
                    # Extract operating expenses
                    operating_expenses_value = income_statement.get('operating_expenses', 0)
                    income_data['operating_expenses'].append({
                        'period': period_key,
                        'value': operating_expenses_value
                    })
                    
                    # Extract operating income
                    operating_income_value = income_statement.get('operating_income', 0)
                    income_data['operating_income'].append({
                        'period': period_key,
                        'value': operating_income_value
                    })
                    
                    # Extract net income
                    net_value = income_statement.get('net_income', 0)
                    income_data['net_income'].append({
                        'period': period_key,
                        'value': net_value
                    })
                    
                    # Extract EPS
                    eps_value = income_statement.get('eps_diluted', 0) or income_statement.get('eps_basic', 0)
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
                period_key = qd.period_key
                
                # Get balance sheet data directly from the new structure
                if hasattr(qd.financial_data, 'balance_sheet') and qd.financial_data.balance_sheet:
                    balance_sheet = qd.financial_data.balance_sheet
                    
                    # Extract total assets
                    assets_value = balance_sheet.get('total_assets', 0)
                    balance_data['total_assets'].append({
                        'period': period_key,
                        'value': assets_value
                    })
                    
                    # Extract current assets
                    current_assets_value = balance_sheet.get('current_assets', 0)
                    balance_data['current_assets'].append({
                        'period': period_key,
                        'value': current_assets_value
                    })
                    
                    # Extract total liabilities
                    liabilities_value = balance_sheet.get('total_liabilities', 0)
                    balance_data['total_liabilities'].append({
                        'period': period_key,
                        'value': liabilities_value
                    })
                    
                    # Extract current liabilities
                    current_liabilities_value = balance_sheet.get('current_liabilities', 0)
                    balance_data['current_liabilities'].append({
                        'period': period_key,
                        'value': current_liabilities_value
                    })
                    
                    # Extract shareholders equity
                    equity_value = balance_sheet.get('stockholders_equity', 0)
                    balance_data['shareholders_equity'].append({
                        'period': period_key,
                        'value': equity_value
                    })
                    
                    # Extract cash and equivalents
                    cash_value = balance_sheet.get('cash', 0)
                    balance_data['cash_and_equivalents'].append({
                        'period': period_key,
                        'value': cash_value
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
                period_key = qd.period_key
                
                # Get cash flow data directly from the new structure
                if hasattr(qd.financial_data, 'cash_flow_statement') and qd.financial_data.cash_flow_statement:
                    cash_flow = qd.financial_data.cash_flow_statement
                    
                    # Extract operating cash flow
                    operating_value = cash_flow.get('operating_cash_flow', 0)
                    cash_flow_data['operating_cash_flow'].append({
                        'period': period_key,
                        'value': operating_value
                    })
                    
                    # Extract investing cash flow
                    investing_value = cash_flow.get('investing_cash_flow', 0)
                    cash_flow_data['investing_cash_flow'].append({
                        'period': period_key,
                        'value': investing_value
                    })
                    
                    # Extract financing cash flow
                    financing_value = cash_flow.get('financing_cash_flow', 0)
                    cash_flow_data['financing_cash_flow'].append({
                        'period': period_key,
                        'value': financing_value
                    })
                    
                    # Calculate free cash flow (operating - capex) if available
                    # For now, we'll use operating cash flow as a proxy
                    free_cash_flow_value = operating_value  # Could subtract capex if available
                    cash_flow_data['free_cash_flow'].append({
                        'period': period_key,
                        'value': free_cash_flow_value
                    })
                    
                    # For now, we don't have capex separately, so use 0
                    cash_flow_data['capital_expenditures'].append({
                        'period': period_key,
                        'value': 0
                    })
            
            # Calculate metrics
            cash_flow_data['metrics'] = self._calculate_cash_flow_metrics(cash_flow_data)
            
            return cash_flow_data
            
        except Exception as e:
            self.main_logger.error(f"Error aggregating cash flow: {e}")
            return {}
    
    def _get_financial_data_dict(self, financial_data) -> Dict[str, Any]:
        """Convert FinancialStatementData object to dictionary format for aggregation"""
        try:
            # If financial_data is already a dict, return it
            if isinstance(financial_data, dict):
                return financial_data
            
            # If it's a FinancialStatementData object, extract the comprehensive data
            if hasattr(financial_data, 'comprehensive_data') and financial_data.comprehensive_data:
                return financial_data.comprehensive_data
            
            # Fallback: try to build from individual fields
            result = {}
            
            if hasattr(financial_data, 'quarterly_data') and financial_data.quarterly_data:
                result.update(financial_data.quarterly_data)
            
            # Add structured statement data if available
            if hasattr(financial_data, 'income_statement') and financial_data.income_statement:
                result.update(financial_data.income_statement)
            
            if hasattr(financial_data, 'balance_sheet') and financial_data.balance_sheet:
                result.update(financial_data.balance_sheet)
            
            if hasattr(financial_data, 'cash_flow_statement') and financial_data.cash_flow_statement:
                result.update(financial_data.cash_flow_statement)
            
            return result
            
        except Exception as e:
            self.main_logger.error(f"Error converting financial data to dict: {e}")
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
                period_key = qd.period_key
                
                # Get financial data directly from the new structure
                if hasattr(qd.financial_data, 'income_statement') and qd.financial_data.income_statement:
                    income_statement = qd.financial_data.income_statement
                    balance_sheet = qd.financial_data.balance_sheet if hasattr(qd.financial_data, 'balance_sheet') else {}
                    
                    # Get key values for ratio calculations
                    revenue = income_statement.get('revenue', 0)
                    net_income = income_statement.get('net_income', 0)
                    total_assets = balance_sheet.get('total_assets', 0)
                    current_assets = balance_sheet.get('current_assets', 0)
                    current_liabilities = balance_sheet.get('current_liabilities', 0)
                    stockholders_equity = balance_sheet.get('stockholders_equity', 0)
                    total_liabilities = balance_sheet.get('total_liabilities', 0)
                    
                    # Calculate profitability ratios
                    if revenue and revenue != 0:
                        net_margin = (net_income / revenue) * 100 if net_income else 0
                        ratios['profitability'][period_key] = {
                            'net_profit_margin': net_margin
                        }
                    
                    # Calculate efficiency ratios
                    if total_assets and total_assets != 0:
                        roa = (net_income / total_assets) * 100 if net_income else 0
                        if 'efficiency' not in ratios:
                            ratios['efficiency'] = {}
                        ratios['efficiency'][period_key] = {
                            'return_on_assets': roa
                        }
                    
                    # Calculate leverage ratios  
                    if stockholders_equity and stockholders_equity != 0:
                        roe = (net_income / stockholders_equity) * 100 if net_income else 0
                        debt_to_equity = (total_liabilities / stockholders_equity) if total_liabilities else 0
                        ratios['leverage'][period_key] = {
                            'return_on_equity': roe,
                            'debt_to_equity': debt_to_equity
                        }
                    
                    # Calculate liquidity ratios
                    if current_liabilities and current_liabilities != 0:
                        current_ratio = (current_assets / current_liabilities) if current_assets else 0
                        ratios['liquidity'][period_key] = {
                            'current_ratio': current_ratio
                        }
            
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
                
                # Get financial data directly from the new structure
                current_income = current.financial_data.income_statement if hasattr(current.financial_data, 'income_statement') else {}
                previous_income = previous.financial_data.income_statement if hasattr(previous.financial_data, 'income_statement') else {}
                
                # Revenue growth
                current_revenue = current_income.get('revenue', 0)
                previous_revenue = previous_income.get('revenue', 0)
                
                if current_revenue and previous_revenue and previous_revenue != 0:
                    revenue_growth = ((current_revenue - previous_revenue) / previous_revenue) * 100
                    trends['revenue_growth'].append({
                        'period': current.period_key,
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
        """Assess the quality and completeness of financial data using domain knowledge"""
        try:
            quality = {
                'completeness_score': 0,
                'core_metrics_score': 0,
                'critical_missing': [],
                'data_issues': [],
                'recommendations': [],
                'period_analysis': []
            }
            
            # Define critical metrics by importance tier
            critical_metrics = {
                'tier_1_essential': ['revenue', 'net_income', 'total_assets', 'stockholders_equity'],
                'tier_2_important': ['operating_income', 'current_assets', 'current_liabilities', 'operating_cash_flow'],
                'tier_3_useful': ['gross_profit', 'total_liabilities', 'cash', 'investing_cash_flow'],
                'tier_4_optional': ['eps_basic', 'eps_diluted', 'accounts_receivable', 'financing_cash_flow'],
                'tier_5_industry_specific': ['inventory', 'long_term_debt', 'cost_of_revenue', 'rd_expense']
            }
            
            # Weights for each tier
            tier_weights = {
                'tier_1_essential': 1.0,      # 100% weight - must have
                'tier_2_important': 0.8,      # 80% weight - should have
                'tier_3_useful': 0.6,         # 60% weight - nice to have
                'tier_4_optional': 0.3,       # 30% weight - bonus
                'tier_5_industry_specific': 0.1  # 10% weight - context dependent
            }
            
            total_periods = len(quarterly_data)
            period_scores = []
            
            for qd in quarterly_data:
                period_analysis = {
                    'period': qd.period_key,
                    'form_type': qd.form_type,
                    'tier_scores': {},
                    'missing_critical': [],
                    'period_score': 0
                }
                
                # Get all metrics for this period
                all_metrics = {}
                if hasattr(qd.financial_data, 'income_statement'):
                    all_metrics.update(qd.financial_data.income_statement)
                if hasattr(qd.financial_data, 'balance_sheet'):
                    all_metrics.update(qd.financial_data.balance_sheet)
                if hasattr(qd.financial_data, 'cash_flow_statement'):
                    all_metrics.update(qd.financial_data.cash_flow_statement)
                
                # Evaluate each tier
                weighted_score = 0
                total_weight = 0
                
                for tier, metrics in critical_metrics.items():
                    tier_found = 0
                    tier_total = len(metrics)
                    
                    for metric in metrics:
                        if metric in all_metrics and all_metrics[metric] != 0:
                            tier_found += 1
                        elif tier in ['tier_1_essential', 'tier_2_important']:
                            period_analysis['missing_critical'].append(metric)
                    
                    tier_score = (tier_found / tier_total) if tier_total > 0 else 0
                    period_analysis['tier_scores'][tier] = {
                        'score': tier_score,
                        'found': tier_found,
                        'total': tier_total
                    }
                    
                    # Add to weighted score
                    weight = tier_weights[tier]
                    weighted_score += tier_score * weight
                    total_weight += weight
                
                # Calculate period score
                period_analysis['period_score'] = (weighted_score / total_weight) if total_weight > 0 else 0
                period_scores.append(period_analysis['period_score'])
                quality['period_analysis'].append(period_analysis)
                
                # Track critical missing across all periods
                for missing in period_analysis['missing_critical']:
                    if missing not in quality['critical_missing']:
                        quality['critical_missing'].append(missing)
            
            # Calculate overall scores
            if period_scores:
                quality['completeness_score'] = (sum(period_scores) / len(period_scores)) * 100
                
                # Core metrics score focuses only on tier 1 & 2
                core_scores = []
                for analysis in quality['period_analysis']:
                    tier1_score = analysis['tier_scores']['tier_1_essential']['score']
                    tier2_score = analysis['tier_scores']['tier_2_important']['score']
                    core_score = (tier1_score * 0.7 + tier2_score * 0.3)  # Weight tier 1 more heavily
                    core_scores.append(core_score)
                
                quality['core_metrics_score'] = (sum(core_scores) / len(core_scores)) * 100
            
            # Generate intelligent recommendations
            self._generate_quality_recommendations(quality)
            
            return quality
            
        except Exception as e:
            self.main_logger.error(f"Error assessing data quality: {e}")
            return {'completeness_score': 0, 'error': str(e)}
    
    def _generate_quality_recommendations(self, quality: Dict[str, Any]) -> None:
        """Generate intelligent recommendations based on data quality analysis"""
        
        core_score = quality['core_metrics_score']
        overall_score = quality['completeness_score']
        critical_missing = quality['critical_missing']
        
        # Core metrics assessment
        if core_score >= 90:
            quality['recommendations'].append("✅ Excellent data quality - all critical metrics available")
        elif core_score >= 75:
            quality['recommendations'].append("✅ Good data quality - most critical metrics available")
        elif core_score >= 50:
            quality['recommendations'].append("⚠️ Moderate data quality - some critical metrics missing")
        else:
            quality['recommendations'].append("❌ Poor data quality - many critical metrics missing")
        
        # Specific missing metric guidance
        if 'revenue' in critical_missing:
            quality['recommendations'].append("🚨 Critical: Revenue data missing - verify company has filed recent reports")
        if 'net_income' in critical_missing:
            quality['recommendations'].append("🚨 Critical: Net income missing - essential for profitability analysis")
        if 'total_assets' in critical_missing:
            quality['recommendations'].append("🚨 Critical: Total assets missing - balance sheet analysis incomplete")
        if 'stockholders_equity' in critical_missing:
            quality['recommendations'].append("🚨 Critical: Stockholders equity missing - solvency analysis incomplete")
        
        # Industry-specific guidance
        has_inventory = any('inventory' not in analysis['missing_critical'] 
                           for analysis in quality['period_analysis'])
        if not has_inventory:
            quality['recommendations'].append("ℹ️ Inventory data missing (common for service/software companies)")
        
        # Filing type analysis
        form_types = [analysis['form_type'] for analysis in quality['period_analysis']]
        if '10-K' not in form_types:
            quality['recommendations'].append("📋 Consider including annual (10-K) filings for comprehensive analysis")
        
        # Overall assessment
        if overall_score >= 80:
            quality['recommendations'].append("🎯 Data quality sufficient for comprehensive fundamental analysis")
        elif overall_score >= 60:
            quality['recommendations'].append("🎯 Data quality adequate for basic fundamental analysis")
        else:
            quality['recommendations'].append("🎯 Data quality may limit analysis depth - consider additional data sources")
    
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