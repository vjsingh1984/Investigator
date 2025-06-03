#!/usr/bin/env python3
"""
InvestiGator - Core Design Pattern Interfaces
Copyright (c) 2025 Vijaykumar Singh
Licensed under the Apache License 2.0

Design Pattern Interfaces and Abstract Classes
Defines the contracts for the OOP-based fundamental analysis system
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Protocol
from dataclasses import dataclass
from enum import Enum
import logging

# ============================================================================
# Data Models
# ============================================================================

@dataclass
class AnalysisRequest:
    """Request object for analysis operations"""
    symbol: str
    cik: Optional[str] = None
    max_periods: int = 4
    include_amendments: bool = True
    force_refresh: bool = False
    analysis_type: str = "comprehensive"

@dataclass
class AnalysisResult:
    """Result object for analysis operations"""
    symbol: str
    financial_health_score: float
    business_quality_score: float
    growth_prospects_score: float
    overall_score: float
    key_insights: List[str]
    key_risks: List[str]
    confidence_level: str
    analysis_summary: str
    metadata: Dict[str, Any]

@dataclass
class QuarterlyMetrics:
    """Quarterly financial metrics data"""
    symbol: str
    fiscal_year: int
    fiscal_period: str
    form_type: str
    filing_date: str
    accession_number: str
    financial_data: Dict[str, Any]
    data_quality_score: float = 0.0

class DataSourceType(Enum):
    """Types of data sources"""
    SEC_SUBMISSIONS = "sec_submissions"
    SEC_COMPANY_FACTS = "sec_company_facts"
    SEC_FRAME_API = "sec_frame_api"
    YAHOO_FINANCE = "yahoo_finance"
    DATABASE_CACHE = "database_cache"

class AnalysisStrategy(Enum):
    """Analysis strategy types"""
    COMPREHENSIVE = "comprehensive"
    FUNDAMENTAL_ONLY = "fundamental_only"
    TECHNICAL_ONLY = "technical_only"
    QUICK_ASSESSMENT = "quick_assessment"

# ============================================================================
# Observer Pattern Interfaces
# ============================================================================

class IAnalysisObserver(ABC):
    """Observer interface for analysis progress tracking"""
    
    @abstractmethod
    def on_analysis_started(self, request: AnalysisRequest) -> None:
        """Called when analysis starts"""
        pass
    
    @abstractmethod
    def on_data_fetched(self, source: DataSourceType, symbol: str, data_size: int) -> None:
        """Called when data is fetched from a source"""
        pass
    
    @abstractmethod
    def on_analysis_progress(self, symbol: str, stage: str, progress: float) -> None:
        """Called to report analysis progress"""
        pass
    
    @abstractmethod
    def on_analysis_completed(self, result: AnalysisResult) -> None:
        """Called when analysis completes"""
        pass
    
    @abstractmethod
    def on_analysis_error(self, symbol: str, error: Exception) -> None:
        """Called when analysis encounters an error"""
        pass

class IAnalysisSubject(ABC):
    """Subject interface for observer pattern"""
    
    @abstractmethod
    def attach_observer(self, observer: IAnalysisObserver) -> None:
        """Attach an observer"""
        pass
    
    @abstractmethod
    def detach_observer(self, observer: IAnalysisObserver) -> None:
        """Detach an observer"""
        pass
    
    @abstractmethod
    def notify_observers(self, event_type: str, **kwargs) -> None:
        """Notify all observers of an event"""
        pass

# ============================================================================
# Strategy Pattern Interfaces
# ============================================================================

class IAnalysisStrategy(ABC):
    """Strategy interface for different analysis approaches"""
    
    @abstractmethod
    def analyze(self, quarterly_data: List[QuarterlyMetrics]) -> AnalysisResult:
        """Perform analysis using this strategy"""
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get the name of this strategy"""
        pass
    
    @abstractmethod
    def get_required_data_sources(self) -> List[DataSourceType]:
        """Get required data sources for this strategy"""
        pass

class IDataAggregationStrategy(ABC):
    """Strategy interface for data aggregation methods"""
    
    @abstractmethod
    def aggregate(self, quarterly_data: List[QuarterlyMetrics]) -> Dict[str, Any]:
        """Aggregate quarterly data using this strategy"""
        pass

class IScoringStrategy(ABC):
    """Strategy interface for scoring methodologies"""
    
    @abstractmethod
    def calculate_scores(self, aggregated_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate financial scores using this strategy"""
        pass

# ============================================================================
# Factory Pattern Interfaces
# ============================================================================

class IDataProcessor(ABC):
    """Abstract interface for data processors"""
    
    @abstractmethod
    def process(self, symbol: str, **kwargs) -> List[QuarterlyMetrics]:
        """Process data for a symbol"""
        pass
    
    @abstractmethod
    def get_supported_source_type(self) -> DataSourceType:
        """Get the data source type this processor supports"""
        pass
    
    @abstractmethod
    def validate_input(self, symbol: str, **kwargs) -> bool:
        """Validate input parameters"""
        pass

class IDataProcessorFactory(ABC):
    """Factory interface for creating data processors"""
    
    @abstractmethod
    def create_processor(self, source_type: DataSourceType) -> IDataProcessor:
        """Create a data processor for the specified source type"""
        pass
    
    @abstractmethod
    def get_supported_types(self) -> List[DataSourceType]:
        """Get list of supported data source types"""
        pass

# ============================================================================
# Adapter Pattern Interfaces
# ============================================================================

class IExternalDataSource(ABC):
    """Interface for external data sources"""
    
    @abstractmethod
    def fetch_data(self, identifier: str, **params) -> Dict[str, Any]:
        """Fetch data from external source"""
        pass
    
    @abstractmethod
    def get_source_name(self) -> str:
        """Get the name of this data source"""
        pass

class IDataSourceAdapter(ABC):
    """Adapter interface for external data sources"""
    
    @abstractmethod
    def adapt_data(self, raw_data: Dict[str, Any], symbol: str) -> List[QuarterlyMetrics]:
        """Adapt external data to internal format"""
        pass
    
    @abstractmethod
    def get_adapted_source_type(self) -> DataSourceType:
        """Get the source type this adapter handles"""
        pass

# ============================================================================
# Builder Pattern Interfaces
# ============================================================================

class IAnalysisConfigBuilder(ABC):
    """Builder interface for analysis configuration"""
    
    @abstractmethod
    def set_symbol(self, symbol: str) -> 'IAnalysisConfigBuilder':
        """Set the symbol to analyze"""
        pass
    
    @abstractmethod
    def set_strategy(self, strategy: AnalysisStrategy) -> 'IAnalysisConfigBuilder':
        """Set the analysis strategy"""
        pass
    
    @abstractmethod
    def set_max_periods(self, periods: int) -> 'IAnalysisConfigBuilder':
        """Set maximum periods to analyze"""
        pass
    
    @abstractmethod
    def add_data_source(self, source: DataSourceType) -> 'IAnalysisConfigBuilder':
        """Add a data source"""
        pass
    
    @abstractmethod
    def add_observer(self, observer: IAnalysisObserver) -> 'IAnalysisConfigBuilder':
        """Add an observer"""
        pass
    
    @abstractmethod
    def build(self) -> AnalysisRequest:
        """Build the analysis request"""
        pass

# ============================================================================
# Repository Pattern Interface
# ============================================================================

class IAnalysisRepository(ABC):
    """Repository interface for analysis data persistence"""
    
    @abstractmethod
    def save_analysis(self, result: AnalysisResult) -> bool:
        """Save analysis result"""
        pass
    
    @abstractmethod
    def get_analysis(self, symbol: str) -> Optional[AnalysisResult]:
        """Get latest analysis for symbol"""
        pass
    
    @abstractmethod
    def get_analysis_history(self, symbol: str, limit: int = 10) -> List[AnalysisResult]:
        """Get analysis history for symbol"""
        pass
    
    @abstractmethod
    def delete_analysis(self, symbol: str) -> bool:
        """Delete analysis for symbol"""
        pass

# ============================================================================
# Chain of Responsibility Pattern Interface
# ============================================================================

class IDataValidator(ABC):
    """Chain of responsibility interface for data validation"""
    
    def __init__(self):
        self._next_validator: Optional['IDataValidator'] = None
    
    def set_next(self, validator: 'IDataValidator') -> 'IDataValidator':
        """Set the next validator in the chain"""
        self._next_validator = validator
        return validator
    
    @abstractmethod
    def validate(self, data: QuarterlyMetrics) -> bool:
        """Validate the data and pass to next validator if valid"""
        pass
    
    def _pass_to_next(self, data: QuarterlyMetrics) -> bool:
        """Pass validation to next validator in chain"""
        if self._next_validator:
            return self._next_validator.validate(data)
        return True

# ============================================================================
# Command Pattern Interface
# ============================================================================

class IAnalysisCommand(ABC):
    """Command interface for analysis operations"""
    
    @abstractmethod
    def execute(self) -> AnalysisResult:
        """Execute the analysis command"""
        pass
    
    @abstractmethod
    def undo(self) -> bool:
        """Undo the analysis command if possible"""
        pass
    
    @abstractmethod
    def get_command_name(self) -> str:
        """Get the name of this command"""
        pass

# ============================================================================
# Facade Pattern Interface
# ============================================================================

class IFundamentalAnalysisFacade(ABC):
    """Facade interface for simplified fundamental analysis"""
    
    @abstractmethod
    def analyze_symbol(self, symbol: str, **options) -> AnalysisResult:
        """Simplified interface for symbol analysis"""
        pass
    
    @abstractmethod
    def analyze_portfolio(self, symbols: List[str], **options) -> Dict[str, AnalysisResult]:
        """Simplified interface for portfolio analysis"""
        pass
    
    @abstractmethod
    def get_analysis_summary(self, symbol: str) -> Dict[str, Any]:
        """Get simplified analysis summary"""
        pass

# ============================================================================
# Template Method Pattern Interface
# ============================================================================

class IAnalysisTemplate(ABC):
    """Template method interface for analysis workflow"""
    
    def analyze(self, request: AnalysisRequest) -> AnalysisResult:
        """Template method defining the analysis workflow"""
        try:
            # Template method implementation
            self._validate_request(request)
            data = self._fetch_data(request)
            validated_data = self._validate_data(data)
            aggregated_data = self._aggregate_data(validated_data)
            scores = self._calculate_scores(aggregated_data)
            insights = self._generate_insights(aggregated_data, scores)
            return self._create_result(request, scores, insights, aggregated_data)
        except Exception as e:
            return self._handle_error(request, e)
    
    @abstractmethod
    def _validate_request(self, request: AnalysisRequest) -> None:
        """Validate the analysis request"""
        pass
    
    @abstractmethod
    def _fetch_data(self, request: AnalysisRequest) -> List[QuarterlyMetrics]:
        """Fetch quarterly data"""
        pass
    
    @abstractmethod
    def _validate_data(self, data: List[QuarterlyMetrics]) -> List[QuarterlyMetrics]:
        """Validate fetched data"""
        pass
    
    @abstractmethod
    def _aggregate_data(self, data: List[QuarterlyMetrics]) -> Dict[str, Any]:
        """Aggregate quarterly data"""
        pass
    
    @abstractmethod
    def _calculate_scores(self, aggregated_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate financial scores"""
        pass
    
    @abstractmethod
    def _generate_insights(self, aggregated_data: Dict[str, Any], scores: Dict[str, float]) -> Dict[str, List[str]]:
        """Generate insights and risks"""
        pass
    
    @abstractmethod
    def _create_result(self, request: AnalysisRequest, scores: Dict[str, float], 
                      insights: Dict[str, List[str]], aggregated_data: Dict[str, Any]) -> AnalysisResult:
        """Create final analysis result"""
        pass
    
    @abstractmethod
    def _handle_error(self, request: AnalysisRequest, error: Exception) -> AnalysisResult:
        """Handle analysis errors"""
        pass