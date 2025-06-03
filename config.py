#!/usr/bin/env python3
"""
InvestiGator - Configuration Management Module
Copyright (c) 2025 Vijaykumar Singh
Licensed under the Apache License 2.0

Common Configuration Module
Handles all configuration settings for the Investment AI system
"""

import os
import json
import logging
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum

@dataclass
class DatabaseConfig:
    """Database configuration"""
    host: str
    port: int
    database: str
    username: str
    password: str
    pool_size: int
    max_overflow: int
    
    @property
    def url(self) -> str:
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
@dataclass
class OllamaConfig:
    """Ollama AI configuration"""
    base_url: str
    models: Dict[str, str]
    timeout: int
    max_retries: int
    min_context_size: int
    num_llm_threads: int
    num_predict: Dict[str, int]
    model_info_cache: Dict[str, Dict[str, int]] = field(default_factory=dict)  # Cache for model info
    
@dataclass
class SECConfig:
    """SEC EDGAR API configuration"""
    user_agent: str
    base_url: str
    rate_limit: int
    cache_dir: str
    ticker_cache_file: str
    max_retries: int
    timeout: int
    frame_api_concepts: dict
    frame_api_details: dict
    xbrl_tag_abbreviations: dict
    
    # Submission and period configuration
    require_submissions: bool
    max_periods_to_analyze: int
    include_amended_filings: bool

@dataclass
class EmailConfig:
    """Email configuration"""
    enabled: bool
    smtp_server: str
    smtp_port: int
    username: str
    password: str
    from_address: str
    recipients: List[str]
    use_tls: bool

@dataclass
class AnalysisConfig:
    """Analysis parameters"""
    fundamental_weight: float
    technical_weight: float
    min_score_for_buy: float
    max_score_for_sell: float
    lookback_days: int
    min_volume: int

@dataclass
class LoggingConfig:
    """Consolidated logging configuration"""
    # Symbol-specific logging
    symbol_log_max_bytes: int
    symbol_log_backup_count: int
    symbol_log_format: str
    
    # Main consolidated logging
    main_log_max_bytes: int
    main_log_backup_count: int
    main_log_format: str
    main_log_file: str

class CacheStorageType(Enum):
    """Cache storage types"""
    DISK = "disk"
    RDBMS = "rdbms"
    DISABLED = "disabled"

@dataclass
class CacheConfig:
    """Cache configuration for a specific data type"""
    enabled: bool = True
    storage_type: CacheStorageType = CacheStorageType.DISK
    priority: int = 1  # Negative priority means write-only (no retrieval)
    
    # Disk cache settings
    disk_path: Optional[str] = None
    filename_pattern: str = "{symbol}_{data_type}.json"
    
    # RDBMS cache settings  
    table_name: Optional[str] = None
    key_column: str = "symbol"
    data_column: str = "data"
    
    # TTL and cleanup settings
    ttl_hours: Optional[int] = None  # None means no expiration
    max_entries: Optional[int] = None  # Max entries before cleanup
    
    def __post_init__(self):
        """Validate cache configuration"""
        if self.enabled:
            if self.storage_type == CacheStorageType.DISK and not self.disk_path:
                raise ValueError("disk_path required for disk cache")
            elif self.storage_type == CacheStorageType.RDBMS and not self.table_name:
                raise ValueError("table_name required for RDBMS cache")

@dataclass
class CacheControlConfig:
    """
    Global cache control configuration using list-based approach
    
    Configuration:
    - storage: List of enabled storage backends ["disk", "rdbms"]
        Empty list = no caching
        ["disk"] = disk only
        ["rdbms"] = database only
        ["disk", "rdbms"] = both (default)
        
    - types: List of enabled cache types ["sec", "llm", "technical", "submission", "company_facts", "quarterly_metrics"]
        Empty list = no caching
        Specific types = only those types enabled
        If not specified, all types are enabled
    """
    storage: List[str] = field(default_factory=lambda: ["disk", "rdbms"])
    types: Optional[List[str]] = None  # None means all types enabled
    
    # Cache behavior
    read_from_cache: bool = True  # Allow reading from cache
    write_to_cache: bool = True  # Allow writing to cache
    force_refresh: bool = False  # If True, ignore cache and fetch fresh data
    force_refresh_symbols: Optional[List[str]] = None  # Force refresh for specific symbols
    cache_ttl_override: Optional[int] = None  # Override TTL in hours for all caches
    
    @property
    def use_cache(self) -> bool:
        """Check if caching is enabled at all"""
        return bool(self.storage) and (self.types is None or bool(self.types))
    
    @property
    def use_disk_cache(self) -> bool:
        """Check if disk caching is enabled"""
        return "disk" in self.storage
    
    @property
    def use_rdbms_cache(self) -> bool:
        """Check if RDBMS caching is enabled"""
        return "rdbms" in self.storage
    
    def is_cache_type_enabled(self, cache_type: str) -> bool:
        """Check if a specific cache type is enabled"""
        if not self.use_cache:
            return False
            
        # If types not specified, all are enabled
        if self.types is None:
            return True
            
        # Map from CacheType enum values to config strings
        type_map = {
            "sec_response": "sec",
            "llm_response": "llm", 
            "technical_data": "technical",
            "submission_data": "submission",
            "company_facts": "company_facts",
            "quarterly_metrics": "quarterly_metrics"
        }
        
        # Also support direct names
        direct_map = {
            "sec": "sec",
            "llm": "llm",
            "technical": "technical",
            "submission": "submission",
            "company_facts": "company_facts",
            "quarterly_metrics": "quarterly_metrics"
        }
        
        config_name = type_map.get(cache_type, direct_map.get(cache_type, cache_type))
        return config_name in self.types

@dataclass
class ParquetConfig:
    """Parquet storage configuration - uniform gzip compression for all data"""
    engine: str = "fastparquet"  # Options: "fastparquet", "pyarrow"
    compression: str = "gzip"  # Uniform compression across all data
    compression_level: Optional[int] = 9  # Maximum compression for gzip
    
    # PyArrow specific options (also use gzip for uniformity)
    pyarrow_compression: str = "gzip"  # Uniform gzip compression
    pyarrow_compression_level: Optional[int] = 9  # Maximum compression for gzip
    
    # Common options
    use_dictionary: bool = True  # Dictionary encoding for string columns
    row_group_size: Optional[int] = None  # Number of rows per row group
    
    def get_write_kwargs(self) -> Dict[str, any]:
        """Get kwargs for to_parquet based on engine"""
        if self.engine == "fastparquet":
            return {
                "engine": "fastparquet",
                "compression": self.compression,
                "index": True
            }
        elif self.engine == "pyarrow":
            kwargs = {
                "engine": "pyarrow",
                "compression": self.pyarrow_compression,
                "index": True,
                "use_dictionary": self.use_dictionary
            }
            # Add compression level if applicable
            if self.pyarrow_compression in ["zstd", "gzip", "brotli"]:
                kwargs["compression"] = {
                    "codec": self.pyarrow_compression,
                    "compression_level": self.pyarrow_compression_level
                }
            if self.row_group_size:
                kwargs["row_group_size"] = self.row_group_size
            return kwargs
        else:
            raise ValueError(f"Unknown parquet engine: {self.engine}")

@dataclass 
class CachingConfig:
    """
    Comprehensive caching configuration for all data types.
    
    This class defines cache behaviors for:
    1. SEC API responses (submissions, company facts, consolidated frames)
    2. LLM analysis responses (fundamental, technical, synthesis)
    3. Audit trails (write-only caches for compliance)
    
    Cache Types:
    - DISK: Files stored as {symbol}_{data_type}.json in specified path
    - RDBMS: Records stored in database table with symbol as key
    - DISABLED: No caching
    
    Priority Levels:
    - Positive (≥1): Normal cache with read/write operations
    - Negative (<0): Write-only cache (no retrieval, for audit/backup)
    
    Storage Patterns:
    - SEC submissions: {symbol}_submissions.json (disk)
    - Consolidated frames: {symbol}_{fiscal_year}_{fiscal_period} (RDBMS)
    - LLM responses: {symbol}_{analysis_type}_{period}.json (disk/RDBMS)
    
    TTL (Time To Live):
    - SEC data: 24 hours (changes infrequently)
    - Quarterly data: 168 hours (1 week, stable once filed)
    - LLM analysis: 24-72 hours (may change with model updates)
    - Synthesis reports: 168 hours (relatively stable)
    """
    
    # SEC API Response Caches
    sec_submissions: CacheConfig = field(default_factory=lambda: CacheConfig(
        enabled=True,
        storage_type=CacheStorageType.DISK,
        priority=1,
        disk_path="data/sec_cache",
        filename_pattern="{symbol}_submissions.json",
        ttl_hours=24  # SEC submissions change infrequently
    ))
    
    sec_company_facts: CacheConfig = field(default_factory=lambda: CacheConfig(
        enabled=True,
        storage_type=CacheStorageType.DISK,
        priority=1,
        disk_path="data/sec_cache",
        filename_pattern="{symbol}_companyfacts.json",
        ttl_hours=24
    ))
    
    sec_consolidated_frames: CacheConfig = field(default_factory=lambda: CacheConfig(
        enabled=True,
        storage_type=CacheStorageType.RDBMS,
        priority=1,
        table_name="sec_consolidated_frames_cache",
        key_column="cache_key",  # Format: {symbol}_{fiscal_year}_{fiscal_period}
        data_column="consolidated_data",
        ttl_hours=168  # 1 week - quarterly data is stable
    ))
    
    # LLM Response Caches
    llm_sec_analysis: CacheConfig = field(default_factory=lambda: CacheConfig(
        enabled=True,
        storage_type=CacheStorageType.DISK,
        priority=1,
        disk_path="data/llm_cache",
        filename_pattern="{symbol}_sec_analysis_{period}.json",
        ttl_hours=72  # 3 days - analysis may change with model updates
    ))
    
    llm_technical_analysis: CacheConfig = field(default_factory=lambda: CacheConfig(
        enabled=True,
        storage_type=CacheStorageType.DISK, 
        priority=1,
        disk_path="data/llm_cache",
        filename_pattern="{symbol}_technical_analysis.json",
        ttl_hours=24  # 1 day - technical analysis changes frequently
    ))
    
    llm_synthesis_reports: CacheConfig = field(default_factory=lambda: CacheConfig(
        enabled=True,
        storage_type=CacheStorageType.RDBMS,
        priority=1,
        table_name="llm_synthesis_cache",
        key_column="report_key",  # Format: {symbol}_{analysis_date}
        data_column="synthesis_data",
        ttl_hours=168  # 1 week - final reports are relatively stable
    ))
    
    # Write-only caches (priority < 0) for audit/backup purposes
    audit_sec_requests: CacheConfig = field(default_factory=lambda: CacheConfig(
        enabled=True,
        storage_type=CacheStorageType.RDBMS,
        priority=-1,  # Write-only, no retrieval
        table_name="sec_api_requests_audit",
        key_column="request_id",
        data_column="request_data",
        max_entries=10000  # Keep last 10k requests
    ))
    
    audit_llm_interactions: CacheConfig = field(default_factory=lambda: CacheConfig(
        enabled=True,
        storage_type=CacheStorageType.RDBMS,
        priority=-1,  # Write-only, no retrieval
        table_name="llm_interactions_audit",
        key_column="interaction_id",
        data_column="interaction_data",
        max_entries=50000  # Keep last 50k interactions
    ))
    
    # Performance and behavior settings
    cache_manager_settings: Dict[str, Union[str, int, float, bool]] = field(default_factory=lambda: {
        "default_timeout_seconds": 30,
        "max_concurrent_operations": 10,
        "compression_enabled": True,
        "compression_threshold_bytes": 1024,  # Compress if > 1KB
        "background_cleanup_enabled": True,
        "cleanup_interval_hours": 6,
        "metrics_collection_enabled": True,
        "cache_hit_ratio_target": 0.85
    })
    
    def get_cache_config(self, data_type: str) -> Optional[CacheConfig]:
        """Get cache configuration for a specific data type"""
        return getattr(self, data_type, None)

class Config:
    """Main configuration class"""
    
    def __init__(self, config_file: str = "config.json"):
        self.config_file = config_file
        self._load_config()
        self._setup_logging()
    
    def _load_config(self):
        """Load configuration from file and environment"""
        # Load from JSON file
        config_data = {}
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                config_data = json.load(f)
        
        # Override with environment variables
        config_data = self._override_with_env(config_data)
        
        # Initialize configuration objects with proper default merging
        # Database config
        database_config = self._default_database()
        database_config.update(config_data.get('database', {}))
        self.database = DatabaseConfig(**database_config)
        
        # Ollama config
        ollama_config = self._default_ollama()
        user_ollama = config_data.get('ollama', {})
        # Deep merge for models - only update if user provides non-empty models
        if 'models' in user_ollama:
            if user_ollama['models']:  # Only update if not empty
                ollama_config['models'].update(user_ollama['models'])
            # Remove models from user_ollama to avoid overwriting with empty dict
            user_ollama = {k: v for k, v in user_ollama.items() if k != 'models'}
        ollama_config.update(user_ollama)
        self.ollama = OllamaConfig(**ollama_config)
        
        # SEC config
        sec_config = self._default_sec()
        sec_config.update(config_data.get('sec', {}))
        self.sec = SECConfig(**sec_config)
        
        # Email config
        email_config = self._default_email()
        email_config.update(config_data.get('email', {}))
        self.email = EmailConfig(**email_config)
        # Analysis config
        analysis_config = self._default_analysis()
        analysis_config.update(config_data.get('analysis', {}))
        self.analysis = AnalysisConfig(**analysis_config)
        
        # Logging config
        logging_config = self._default_logging()
        logging_config.update(config_data.get('logging', {}))
        self.logging = LoggingConfig(**logging_config)
        
        # Caching config
        caching_data = config_data.get('caching', {})
        self.caching = self._build_caching_config(caching_data)
        
        # Cache control config
        cache_control_config = self._default_cache_control()
        cache_control_config.update(config_data.get('cache_control', {}))
        self.cache_control = CacheControlConfig(**cache_control_config)
        
        # Parquet config
        parquet_config = self._default_parquet()
        parquet_config.update(config_data.get('parquet', {}))
        self.parquet = ParquetConfig(**parquet_config)
        
        # Other settings
        self.stocks_to_track = config_data.get('stocks_to_track', self._default_stocks())
        self.data_dir = Path(config_data.get('data_dir', './data'))
        self.reports_dir = Path(config_data.get('reports_dir', './reports'))
        self.logs_dir = Path(config_data.get('logs_dir', './logs'))
        
        # Ensure directories exist
        for directory in [self.data_dir, self.reports_dir, self.logs_dir, self.sec.cache_dir]:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def _override_with_env(self, config_data: Dict) -> Dict:
        """Override configuration with environment variables"""
        env_mappings = {
            'DB_HOST': ('database', 'host'),
            'DB_PORT': ('database', 'port'),
            'DB_NAME': ('database', 'database'),
            'DB_USER': ('database', 'username'),
            'DB_PASSWORD': ('database', 'password'),
            'OLLAMA_URL': ('ollama', 'base_url'),
            'EMAIL_USERNAME': ('email', 'username'),
            'EMAIL_PASSWORD': ('email', 'password'),
            'EMAIL_SMTP_SERVER': ('email', 'smtp_server'),
            'EMAIL_SMTP_PORT': ('email', 'smtp_port'),
        }
        
        for env_var, (section, key) in env_mappings.items():
            if env_var in os.environ:
                if section not in config_data:
                    config_data[section] = {}
                
                value = os.environ[env_var]
                # Convert to appropriate type
                if key == 'port':
                    value = int(value)
                elif key in ['enabled', 'use_tls']:
                    value = value.lower() in ('true', '1', 'yes')
                
                config_data[section][key] = value
        
        return config_data
    
    def _default_database(self) -> Dict:
        """Default database configuration"""
        return {
            'host': 'localhost',
            'port': 5432,
            'database': 'investment_ai',
            'username': 'investment_user',
            'password': 'investment_pass',
            'pool_size': 10,
            'max_overflow': 20
        }
    
    def _default_ollama(self) -> Dict:
        """Default Ollama configuration"""
        return {
            'base_url': 'http://localhost:11434',
            'timeout': 600,
            'max_retries': 3,
            'min_context_size': 8192,
            'num_llm_threads': 1,
            'models': {
                'fundamental_analysis': 'llama3.1:8b-instruct-q8_0',
                'technical_analysis': 'llama3.1:8b-instruct-q8_0',
                'report_generation': 'llama3.1:8b-instruct-q8_0',
                'synthesizer': 'llama3.1:8b-instruct-q8_0'
            },
            'num_predict': {
                'fundamental_analysis': 2048,
                'technical_analysis': 1536,
                'report_generation': 2048,
                'synthesizer': 1536
            }
        }
    
    def _default_sec(self) -> Dict:
        """Default SEC configuration"""
        return {
            'user_agent': 'InvestiGator/1.0 (singhvjd@gmail.com)',
            'base_url': 'https://data.sec.gov',
            'rate_limit': 10,
            'cache_dir': './data/sec_cache',
            'ticker_cache_file': './data/ticker_cik_map.txt',
            'max_retries': 3,
            'timeout': 30,
            # XBRL tag to abbreviation mapping for efficient storage and LLM consumption
            'xbrl_tag_abbreviations': {
                # Income Statement
                'RevenueFromContractWithCustomerExcludingAssessedTax': 'Revenue',
                'NetIncomeLossAvailableToCommonStockholdersBasic': 'NetIncome',
                'OpIncome': 'OpIncome',
                'OperatingIncomeLoss': 'OpIncome',
                'IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest': 'PreTaxIncome',
                'GrossProfit': 'GrossProfit',
                'EPS': 'EPS',
                'EarningsPerShareDiluted': 'EPSDiluted',
                'EarningsPerShareBasic': 'EPSBasic',
                'CostOfGoodsAndServicesSold': 'COGS',
                'CostOfRevenue': 'COGS',
                'SellingGeneralAndAdministrativeExpense': 'SGA',
                'ResearchAndDevelopmentExpense': 'RnD',
                'InterestExp': 'InterestExp',
                'InterestExpense': 'InterestExp',
                'TaxExp': 'TaxExp',
                'IncomeTaxExpenseBenefit': 'TaxExp',
                'DepreciationAndAmortization': 'DA',
                
                # Balance Sheet
                'Assets': 'TotalAssets',
                'AssetsCurrent': 'CurrentAssets',
                'Liabilities': 'TotalLiabilities',
                'LiabilitiesCurrent': 'CurrentLiab',
                'Equity': 'Equity',
                'StockholdersEquity': 'Equity',
                'CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents': 'Cash',
                'Cash': 'Cash',
                'AccountsNotesAndLoansReceivableNetCurrent': 'AR',
                'AccountsReceivableNetCurrent': 'AR',
                'Inventory': 'Inventory',
                'PPE': 'PPE',
                'PropertyPlantAndEquipmentNet': 'PPE',
                'Goodwill': 'Goodwill',
                'IntangibleAssetsNetExcludingGoodwill': 'IntangibleAssets',
                'UnsecuredDebt': 'LongTermDebt',
                'LongTermDebt': 'LongTermDebt',
                'AcctsPayable': 'AP',
                'AccountsPayableCurrent': 'AP',
                'RetainedEarn': 'RetainedEarnings',
                'RetainedEarnings': 'RetainedEarnings',
                'CommonStockValue': 'CommonStock',
                'TreasuryStockCommon': 'TreasuryStock',
                'AdditionalPaidInCapital': 'APIC',
                'AccumulatedOtherComprehensiveIncome': 'AOCI',
                
                # Cash Flow
                'CF_Operations': 'CFO',
                'NetCashProvidedByUsedInOperatingActivities': 'CFO',
                'CF_Investing': 'CFI',
                'NetCashProvidedByUsedInInvestingActivities': 'CFI',
                'CF_Financing': 'CFF',
                'NetCashProvidedByUsedInFinancingActivities': 'CFF',
                'PaymentsToAcquirePropertyPlantAndEquipment': 'CapEx',
                'StockComp': 'StockComp',
                'ShareBasedCompensation': 'StockComp',
                'PaymentsOfDividendsCommonStock': 'DividendsPaid',
                'DividendsCommonStockCash': 'DividendsPaid',
                'PaymentsForRepurchaseOfCommonStock': 'ShareRepurchases',
                'TreasuryStockValueAcquiredCostMethod': 'ShareRepurchases',
                
                # Other/Shares
                'SharesOut': 'SharesOut',
                'SharesOutstanding': 'SharesOut',
                'CommonStockSharesOutstanding': 'SharesOut',
                'SharesBasic': 'SharesBasic',
                'WeightedAverageNumberOfSharesOutstandingBasic': 'SharesBasic',
                'SharesDiluted': 'SharesDiluted',
                'WeightedAverageNumberOfDilutedSharesOutstanding': 'SharesDiluted',
                'EPSDiluted': 'EPSDiluted',
                'CommonStockDividendsPerShareDeclared': 'DividendPerShare',
                'ComprehensiveIncomeNetOfTax': 'ComprehensiveIncome',
                'OtherComprehensiveIncomeLossNetOfTaxPortionAttributableToParent': 'OCI',
                
                # Additional common variations
                'Revenues': 'Revenue',
                'SalesRevenueNet': 'Revenue',
                'NetIncomeLoss': 'NetIncome',
                'ProfitLoss': 'NetIncome',
                'CostOfSales': 'COGS',
                'OperatingExpenses': 'OpExpenses',
                'CostsAndExpenses': 'OpExpenses',
                
                # Text-based disclosures (abbreviated for context efficiency)
                'DocumentAndEntityInformation': 'EntityInfo',
                'EntityRegistrantName': 'CompanyName',
                'EntityCentralIndexKey': 'CIK',
                'ManagementDiscussionAndAnalysis': 'MD&A',
                'DisclosureOfRisksAndUncertainties': 'RiskDisclosure',
                'BusinessDescription': 'BusinessDesc',
                'GeneralBusinessDescription': 'BusinessDesc',
                'RiskFactors': 'Risks',
                'RiskManagement': 'RiskMgmt',
                'SignificantAccountingPolicies': 'AcctPolicies',
                'AccountingPoliciesAndMethods': 'AcctPolicies',
                'SegmentReporting': 'Segments',
                'ReportableSegments': 'Segments',
                'SubsequentEvents': 'SubEvents',
                'EventsOccurringAfterBalanceSheetDate': 'PostBalEvents',
                'CommitmentsAndContingencies': 'Commitments',
                'Contingencies': 'Contingencies'
            },
            'frame_api_concepts': {
                'income_statement': {
                    # Revenue variations (from all_tags.txt)
                    'revenues': ['RevenueFromContractWithCustomerExcludingAssessedTax', 'Revenues', 'SalesRevenueNet', 'RevenueFromContractWithCustomerIncludingAssessedTax', 'Revenue', 'SalesRevenueGoodsNet', 'SalesRevenueServicesNet', 'TotalRevenues', 'NetRevenues', 'RevenueNet', 'RevenuesNetOfInterestExpense'],
                    # Net Income variations 
                    'net_income': ['NetIncomeLoss', 'NetIncomeLossAvailableToCommonStockholdersBasic', 'ProfitLoss', 'NetIncomeLossAttributableToParent', 'NetIncomeLossAttributableToNoncontrollingInterest'],
                    # Operating Income variations
                    'operating_income': ['OperatingIncomeLoss', 'IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest', 'OperatingIncomeLossNoninterestExpense'],
                    # Gross Profit
                    'gross_profit': ['GrossProfit'],
                    # EPS variations
                    'earnings_per_share': ['EarningsPerShareDiluted', 'EarningsPerShareBasic', 'EarningsPerShareBasicAndDiluted'],
                    # Cost variations
                    'cost_of_revenue': ['CostOfGoodsAndServicesSold', 'CostOfRevenue', 'CostOfSales', 'CostOfGoodsSold', 'CostOfGoodsAndServiceExcludingDepreciationDepletionAndAmortization', 'CostOfRevenueExcludingDepreciationDepletionAndAmortization'],
                    # Operating Expenses
                    'operating_expenses': ['OperatingExpenses', 'CostsAndExpenses', 'SellingGeneralAndAdministrativeExpense', 'OperatingCostsAndExpenses'],
                    # R&D
                    'research_development': ['ResearchAndDevelopmentExpense', 'ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost'],
                    # SG&A
                    'selling_general_admin': ['SellingGeneralAndAdministrativeExpense', 'SellingAndMarketingExpense', 'GeneralAndAdministrativeExpense'],
                    # Interest
                    'interest_expense': ['InterestExpense', 'InterestExpenseDebt', 'InterestExpenseDebtExcludingAmortization', 'InterestExpenseNonoperating', 'InterestExpenseNet'],
                    'interest_income': ['InterestIncomeOperating', 'InterestAndOtherIncome', 'InterestAndDividendIncomeOperating'],
                    # Tax
                    'tax_expense': ['IncomeTaxExpenseBenefit', 'CurrentIncomeTaxExpenseBenefit', 'IncomeTaxes'],
                    # Other Income Statement items
                    'other_income': ['OtherNonoperatingIncomeExpense', 'NonoperatingIncomeExpense', 'OtherIncomeExpenseNet'],
                    'depreciation': ['DepreciationAndAmortization', 'DepreciationDepletionAndAmortization', 'DepreciationAmortizationAndAccretionNet'],
                    'production_startup': ['ProductionStartUpExpense', 'StartupCosts']
                },
                'balance_sheet': {
                    # Assets
                    'total_assets': ['Assets'],
                    'current_assets': ['AssetsCurrent', 'AssetsCurrentOther'],
                    'cash_and_equivalents': ['CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents', 'CashAndCashEquivalentsAtCarryingValue', 'CashCashEquivalentsAndShortTermInvestments', 'Cash'],
                    'short_term_investments': ['ShortTermInvestments', 'MarketableSecuritiesCurrent'],
                    'accounts_receivable': ['AccountsReceivableNetCurrent', 'AccountsReceivableNet', 'AccountsNotesAndLoansReceivableNetCurrent'],
                    'inventory': ['InventoryNet', 'Inventory'],
                    'prepaid_expenses': ['PrepaidExpenseAndOtherAssetsCurrent', 'PrepaidExpenseCurrent'],
                    'other_current_assets': ['OtherAssetsCurrent'],
                    'property_plant_equipment': ['PropertyPlantAndEquipmentNet', 'PropertyPlantAndEquipmentGross'],
                    'goodwill': ['Goodwill'],
                    'intangible_assets': ['IntangibleAssetsNetExcludingGoodwill', 'IntangibleAssetsNetIncludingGoodwill'],
                    'long_term_investments': ['LongTermInvestments', 'InvestmentsInAffiliatesSubsidiariesAssociatesAndJointVentures'],
                    'other_assets': ['OtherAssetsNoncurrent', 'AssetsNoncurrentOther'],
                    # Liabilities  
                    'total_liabilities': ['Liabilities', 'LiabilitiesAndStockholdersEquity'],
                    'current_liabilities': ['LiabilitiesCurrent'],
                    'accounts_payable': ['AccountsPayableCurrent', 'AccountsPayableAndAccruedLiabilitiesCurrent'],
                    'accrued_liabilities': ['AccruedLiabilitiesCurrent', 'AccruedIncomeTaxesCurrent'],
                    'short_term_debt': ['ShortTermBorrowings', 'LongTermDebtCurrent'],
                    'other_current_liabilities': ['OtherLiabilitiesCurrent'],
                    'long_term_debt': ['LongTermDebtNoncurrent', 'LongTermDebt', 'LongTermDebtAndCapitalLeaseObligations'],
                    'deferred_tax': ['DeferredTaxLiabilitiesNoncurrent', 'DeferredIncomeTaxLiabilitiesNet'],
                    'other_liabilities': ['OtherLiabilitiesNoncurrent'],
                    # Equity
                    'shareholders_equity': ['StockholdersEquity', 'StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest'],
                    'common_stock': ['CommonStockValue', 'CommonStockParOrStatedValuePerShareValue'],
                    'retained_earnings': ['RetainedEarningsAccumulatedDeficit', 'RetainedEarnings'],
                    'additional_paid_in_capital': ['AdditionalPaidInCapitalCommonStock', 'AdditionalPaidInCapital'],
                    'treasury_stock': ['TreasuryStockValue', 'TreasuryStockCommonValue'],
                    'accumulated_other_comprehensive_income': ['AccumulatedOtherComprehensiveIncomeLossNetOfTax', 'AccumulatedOtherComprehensiveIncome']
                },
                'cash_flow': {
                    # Operating Activities
                    'operating_cash_flow': ['NetCashProvidedByUsedInOperatingActivities', 'NetCashProvidedByUsedInOperatingActivitiesContinuingOperations', 'NetCashProvidedByOperatingActivities', 'NetCashUsedInOperatingActivities', 'CashProvidedByUsedInOperatingActivities'],
                    'depreciation': ['DepreciationDepletionAndAmortization', 'DepreciationAndAmortization', 'Depreciation'],
                    'amortization': ['AmortizationOfIntangibleAssets', 'Amortization'],
                    'stock_based_compensation': ['ShareBasedCompensation', 'StockBasedCompensation'],
                    'deferred_taxes': ['DeferredIncomeTaxExpenseBenefit', 'IncreaseDecreaseInDeferredIncomeTaxes'],
                    'changes_working_capital': ['IncreaseDecreaseInOperatingCapital', 'IncreaseDecreaseInOperatingAssetsAndLiabilities'],
                    'accounts_receivable_changes': ['IncreaseDecreaseInAccountsReceivable'],
                    'inventory_changes': ['IncreaseDecreaseInInventories'],
                    'accounts_payable_changes': ['IncreaseDecreaseInAccountsPayable'],
                    # Investing Activities
                    'investing_cash_flow': ['NetCashProvidedByUsedInInvestingActivities', 'NetCashProvidedByUsedInInvestingActivitiesContinuingOperations'],
                    'capital_expenditures': ['PaymentsToAcquirePropertyPlantAndEquipment', 'PaymentsForCapitalImprovements', 'CapitalExpenditures', 'CapitalExpendituresIncurredButNotYetPaid', 'PaymentsToAcquireProductiveAssets'],
                    'acquisitions': ['PaymentsToAcquireBusinessesNetOfCashAcquired', 'PaymentsToAcquireBusinesses'],
                    'asset_sales': ['ProceedsFromSaleOfPropertyPlantAndEquipment', 'ProceedsFromSaleOfAssets'],
                    'investment_purchases': ['PaymentsToAcquireInvestments', 'PaymentsToAcquireMarketableSecurities'],
                    'investment_sales': ['ProceedsFromSaleOfInvestments', 'ProceedsFromSaleAndMaturityOfMarketableSecurities'],
                    # Financing Activities
                    'financing_cash_flow': ['NetCashProvidedByUsedInFinancingActivities', 'NetCashProvidedByUsedInFinancingActivitiesContinuingOperations'],
                    'dividends_paid': ['PaymentsOfDividendsCommonStock', 'PaymentsOfDividends', 'PaymentsOfOrdinaryDividends'],
                    'stock_repurchases': ['PaymentsForRepurchaseOfCommonStock', 'PaymentsForRepurchaseOfEquity'],
                    'stock_issuance': ['ProceedsFromIssuanceOfCommonStock', 'ProceedsFromStockOptionsExercised'],
                    'debt_issuance': ['ProceedsFromIssuanceOfLongTermDebt', 'ProceedsFromDebtNetOfIssuanceCosts'],
                    'debt_repayment': ['RepaymentsOfLongTermDebt', 'RepaymentsOfDebt'],
                    'other_financing': ['PaymentsForOtherFinancingActivities', 'ProceedsFromOtherFinancingActivities']
                },
                'other': {
                    # Share data
                    'shares_outstanding': ['CommonStockSharesOutstanding', 'SharesOutstanding'],
                    'shares_basic': ['WeightedAverageNumberOfSharesOutstandingBasic', 'CommonStockSharesOutstanding'],
                    'shares_diluted': ['WeightedAverageNumberOfDilutedSharesOutstanding'],
                    'eps_basic': ['EarningsPerShareBasic'],
                    'eps_diluted': ['EarningsPerShareDiluted'],
                    'dividends_per_share': ['CommonStockDividendsPerShareDeclared', 'CommonStockDividendsPerShareCashPaid'],
                    # Comprehensive Income
                    'comprehensive_income': ['ComprehensiveIncomeLossNetOfTax', 'ComprehensiveIncomeLossNetOfTaxIncludingPortionAttributableToNoncontrollingInterest'],
                    'other_comprehensive_income': ['OtherComprehensiveIncomeLossNetOfTaxPortionAttributableToParent', 'OtherComprehensiveIncomeLossNetOfTax'],
                    # Additional metrics (these are typically calculated, not direct XBRL tags)
                    'book_value_per_share': ['BookValuePerShare'],
                    'working_capital': ['WorkingCapital'],
                    'debt_to_equity': ['DebtToEquityRatio'],
                    'current_ratio': ['CurrentRatio'],
                    'quick_ratio': ['QuickRatio'],
                    'gross_margin': ['GrossMargin'],
                    'operating_margin': ['OperatingIncomeLossMargin'],
                    'net_margin': ['ProfitMargin'],
                    'return_on_assets': ['ReturnOnAssets'],
                    'return_on_equity': ['ReturnOnEquity'],
                    # Text-based disclosures (abbreviated for context efficiency)
                    'footnotes_summary': ['DocumentAndEntityInformation', 'EntityRegistrantName', 'EntityCentralIndexKey'],
                    'mgmt_discussion': ['ManagementDiscussionAndAnalysis', 'DisclosureOfRisksAndUncertainties'],
                    'business_description': ['BusinessDescription', 'GeneralBusinessDescription'],
                    'risk_factors': ['RiskFactors', 'RiskManagement'],
                    'accounting_policies': ['SignificantAccountingPolicies', 'AccountingPoliciesAndMethods'],
                    'segment_reporting': ['SegmentReporting', 'ReportableSegments'],
                    'subsequent_events': ['SubsequentEvents', 'EventsOccurringAfterBalanceSheetDate'],
                    'commitments_contingencies': ['CommitmentsAndContingencies', 'Contingencies']
                }
            },
            'frame_api_details': {
                        "income_statement_revenue": {
                                "revenues": ["us-gaap:Revenues", "us-gaap:SalesRevenueNet", "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax", "us-gaap:SalesRevenueServicesNet"],
                                "product_revenue": ["us-gaap:ProductSalesRevenue", "us-gaap:ProductRevenue"],
                                "service_revenue": ["us-gaap:ServiceRevenue", "us-gaap:ServiceSalesRevenue"],
                                "subscription_revenue": ["us-gaap:SubscriptionRevenue", "us-gaap:CloudSubscriptionsAndSupportRevenue"],
                                "licensing_revenue": ["us-gaap:LicensingRevenue", "us-gaap:LicenseRevenue"],
                                "other_revenue": ["us-gaap:OtherRevenue", "us-gaap:RevenuesOther"]
                        },
                        "income_statement_cost_of_revenue": {
                                "cost_of_revenue": ["us-gaap:CostOfRevenue", "us-gaap:CostOfGoodsAndServicesSold", "us-gaap:CostOfGoodsSold"],
                                "cost_of_product_revenue": ["us-gaap:CostOfProductRevenue", "us-gaap:CostOfGoodsSold"],
                                "cost_of_service_revenue": ["us-gaap:CostOfServiceRevenue", "us-gaap:CostOfServices"]
                        },
                        "income_statement_gross_profit": {
                                "gross_profit": ["us-gaap:GrossProfit"]
                        },
                        "income_statement_operating_expenses": {
                                "research_and_development_expense": ["us-gaap:ResearchAndDevelopmentExpense"],
                                "selling_general_and_administrative_expense": ["us-gaap:SellingGeneralAndAdministrativeExpense"],
                                "selling_and_marketing_expense": ["us-gaap:SellingAndMarketingExpense", "us-gaap:MarketingAndAdvertisingExpense"],
                                "general_and_administrative_expense": ["us-gaap:GeneralAndAdministrativeExpense"],
                                "depreciation_expense": ["us-gaap:Depreciation"],
                                "amortization_of_intangible_assets": ["us-gaap:AmortizationOfIntangibleAssets"],
                                "depreciation_and_amortization": ["us-gaap:DepreciationAndAmortization", "us-gaap:DepreciationDepletionAndAmortization"],
                                "restructuring_charges": ["us-gaap:RestructuringCharges", "us-gaap:RestructuringCostsAndOther"],
                                "impairment_charges": ["us-gaap:ImpairmentOfLongLivedAssetsHeldForUse", "us-gaap:GoodwillImpairmentLoss", "us-gaap:AssetImpairmentCharges"]
                        },
                        "income_statement_operating_income": {
                                "operating_income_loss": ["us-gaap:OperatingIncomeLoss"],
                                "other_operating_income_expense": ["us-gaap:OtherOperatingIncomeExpenseNet", "us-gaap:OtherOperatingIncome"]
                        },
                        "income_statement_non_operating_income_expense": {
                                "interest_income": ["us-gaap:InvestmentIncomeInterest", "us-gaap:InterestIncomeOther"],
                                "interest_expense": ["us-gaap:InterestExpense"],
                                "investment_income_loss": ["us-gaap:InvestmentIncomeLoss", "us-gaap:GainsLossesOnInvestments"],
                                "foreign_currency_transaction_gain_loss": ["us-gaap:ForeignCurrencyTransactionGainLossBeforeTax"],
                                "other_nonoperating_income_expense": ["us-gaap:OtherNonoperatingIncomeExpense", "us-gaap:OtherIncomeExpenseNet"]
                        },
                        "income_statement_income_before_tax": {
                                "income_loss_from_continuing_operations_before_income_tax": ["us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxExpenseBenefit", "us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest"]
                        },
                        "income_statement_income_tax": {
                                "income_tax_expense_benefit": ["us-gaap:IncomeTaxExpenseBenefit"],
                                "current_income_tax_expense_benefit": ["us-gaap:CurrentIncomeTaxExpenseBenefit"],
                                "deferred_income_tax_expense_benefit": ["us-gaap:DeferredIncomeTaxExpenseBenefit"]
                        },
                        "income_statement_net_income": {
                                "net_income_loss": ["us-gaap:NetIncomeLoss"],
                                "net_income_loss_attributable_to_parent": ["us-gaap:NetIncomeLossAvailableToCommonStockholdersBasic"],
                                "net_income_loss_from_continuing_operations": ["us-gaap:IncomeLossFromContinuingOperationsNetOfTax"],
                                "net_income_loss_from_discontinued_operations": ["us-gaap:IncomeLossFromDiscontinuedOperationsNetOfTax"],
                                "earnings_per_share_basic": ["us-gaap:EarningsPerShareBasic"],
                                "earnings_per_share_diluted": ["us-gaap:EarningsPerShareDiluted"],
                                "weighted_average_shares_outstanding_basic": ["us-gaap:WeightedAverageNumberOfSharesOutstandingBasic"],
                        "weighted_average_shares_outstanding_diluted": ["us-gaap:WeightedAverageNumberOfDilutedSharesOutstanding"]
                },
                "balance_sheet_current_assets": {
                        "cash_and_cash_equivalents": ["us-gaap:CashAndCashEquivalentsAtCarryingValue"],
                        "short_term_investments": ["us-gaap:MarketableSecuritiesCurrent", "us-gaap:ShortTermInvestments"],
                        "accounts_receivable_net": ["us-gaap:AccountsReceivableNetCurrent"],
                        "inventories_net": ["us-gaap:InventoryNet", "us-gaap:InventoriesNet"],
                        "prepaid_expenses_current": ["us-gaap:PrepaidExpenseCurrent", "us-gaap:PrepaidExpensesAndOtherCurrentAssets"],
                        "other_current_assets": ["us-gaap:OtherCurrentAssets"]
                },
                "balance_sheet_non_current_assets": {
                        "property_plant_and_equipment_net": ["us-gaap:PropertyPlantAndEquipmentNet"],
                        "goodwill": ["us-gaap:Goodwill"],
                        "intangible_assets_net_excluding_goodwill": ["us-gaap:IntangibleAssetsNetExcludingGoodwill", "us-gaap:FiniteLivedIntangibleAssetsNet"],
                        "long_term_investments": ["us-gaap:LongTermInvestments", "us-gaap:MarketableSecuritiesNoncurrent"],
                        "operating_lease_right_of_use_asset": ["us-gaap:OperatingLeaseRightOfUseAsset"],
                        "finance_lease_right_of_use_asset": ["us-gaap:FinanceLeaseRightOfUseAsset"],
                        "deferred_tax_assets_noncurrent": ["us-gaap:DeferredTaxAssetsNetNoncurrent"],
                        "other_non_current_assets": ["us-gaap:OtherAssetsNoncurrent", "us-gaap:OtherNoncurrentAssets"]
                },
                "balance_sheet_current_liabilities": {
                        "accounts_payable_current": ["us-gaap:AccountsPayableCurrent", "us-gaap:AccountsPayableTradeCurrent"],
                        "accrued_liabilities_current": ["us-gaap:AccruedLiabilitiesCurrent", "us-gaap:OtherAccruedLiabilitiesCurrent"],
                        "short_term_debt": ["us-gaap:ShortTermBorrowings", "us-gaap:DebtCurrent", "us-gaap:CommercialPaper"],
                        "current_portion_of_long_term_debt": ["us-gaap:LongTermDebtCurrentMaturities"],
                        "operating_lease_liability_current": ["us-gaap:OperatingLeaseLiabilityCurrent"],
                        "finance_lease_liability_current": ["us-gaap:FinanceLeaseLiabilityCurrent"],
                        "deferred_revenue_current": ["us-gaap:DeferredRevenueCurrent", "us-gaap:ContractWithCustomerLiabilityCurrent"],
                        "income_taxes_payable_current": ["us-gaap:IncomeTaxesPayableCurrent"],
                        "other_current_liabilities": ["us-gaap:OtherCurrentLiabilities"]
                },
                "balance_sheet_non_current_liabilities": {
                        "long_term_debt_noncurrent": ["us-gaap:LongTermDebtNoncurrent", "us-gaap:DebtNoncurrent"],
                        "operating_lease_liability_noncurrent": ["us-gaap:OperatingLeaseLiabilityNoncurrent"],
                        "finance_lease_liability_noncurrent": ["us-gaap:FinanceLeaseLiabilityNoncurrent"],
                        "deferred_tax_liabilities_noncurrent": ["us-gaap:DeferredTaxLiabilitiesNoncurrent"],
                        "pension_and_other_postretirement_benefits_noncurrent": ["us-gaap:PensionAndOtherPostretirementDefinedBenefitPlansLiabilitiesNoncurrent"],
                        "other_non_current_liabilities": ["us-gaap:OtherLiabilitiesNoncurrent", "us-gaap:OtherNoncurrentLiabilities"]
                },
                "balance_sheet_equity": {
                        "common_stock_value": ["us-gaap:CommonStockValue", "us-gaap:CommonStocksIncludingAdditionalPaidInCapital"],
                        "preferred_stock_value": ["us-gaap:PreferredStockValue"],
                        "additional_paid_in_capital": ["us-gaap:AdditionalPaidInCapital"],
                        "retained_earnings_accumulated_deficit": ["us-gaap:RetainedEarningsAccumulatedDeficit"],
                        "accumulated_other_comprehensive_income_loss": ["us-gaap:AccumulatedOtherComprehensiveIncomeLossNetOfTax"],
                        "treasury_stock_value": ["us-gaap:TreasuryStockValue"],
                        "noncontrolling_interest_equity": ["us-gaap:MinorityInterest", "us-gaap:NoncontrollingInterest"]
                },
                "balance_sheet_totals": {
                        "total_assets": ["us-gaap:Assets"],
                        "total_liabilities": ["us-gaap:Liabilities"],
                        "total_equity": ["us-gaap:StockholdersEquity", "us-gaap:PartnersCapital"],
                        "total_liabilities_and_equity": ["us-gaap:LiabilitiesAndStockholdersEquity"]
                },
                "cash_flow_operating": {
                        "net_cash_provided_by_operating_activities": ["us-gaap:NetCashProvidedByUsedInOperatingActivities"],
                        "adjustments_to_reconcile_net_income_to_cash_from_ops_depreciation_amortization": ["us-gaap:DepreciationDepletionAndAmortization"],
                        "adjustments_to_reconcile_net_income_to_cash_from_ops_share_based_compensation": ["us-gaap:ShareBasedCompensation"],
                        "adjustments_to_reconcile_net_income_to_cash_from_ops_deferred_income_tax": ["us-gaap:DeferredIncomeTaxExpenseBenefit"],
                        "change_in_accounts_receivable_cf": ["us-gaap:IncreaseDecreaseInAccountsReceivableTrade"],
                        "change_in_inventories_cf": ["us-gaap:IncreaseDecreaseInInventories"],
                        "change_in_prepaid_expenses_cf": ["us-gaap:IncreaseDecreaseInPrepaidDeferredExpenseAndOtherAssets"],
                        "change_in_accounts_payable_cf": ["us-gaap:IncreaseDecreaseInAccountsPayableTrade"],
                        "change_in_accrued_liabilities_cf": ["us-gaap:IncreaseDecreaseInAccruedLiabilities"],
                        "change_in_deferred_revenue_cf": ["us-gaap:IncreaseDecreaseInDeferredRevenue"]
                },
                "cash_flow_investing": {
                        "net_cash_provided_by_investing_activities": ["us-gaap:NetCashProvidedByUsedInInvestingActivities"],
                        "capital_expenditures": ["us-gaap:PaymentsToAcquirePropertyPlantAndEquipment"],
                        "purchases_of_investments": ["us-gaap:PaymentsToAcquireInvestments"],
                        "sales_maturities_of_investments": ["us-gaap:ProceedsFromSaleAndMaturityOfMarketableSecurities", "us-gaap:ProceedsFromSaleOfMarketableSecurities", "us-gaap:ProceedsFromMaturitiesOfMarketableSecurities"],
                        "acquisitions_net_of_cash_acquired": ["us-gaap:PaymentsToAcquireBusinessesNetOfCashAcquired"],
                        "divestitures_net_of_cash_sold": ["us-gaap:ProceedsFromDivestitureOfBusinessesNetOfCashDivested"]
                },
                "cash_flow_financing": {
                        "net_cash_provided_by_financing_activities": ["us-gaap:NetCashProvidedByUsedInFinancingActivities"],
                        "proceeds_from_debt_issuance": ["us-gaap:ProceedsFromIssuanceOfLongTermDebt"],
                        "repayments_of_debt": ["us-gaap:RepaymentsOfLongTermDebt"],
                        "proceeds_from_equity_issuance": ["us-gaap:ProceedsFromIssuanceOfCommonStock"],
                        "repurchases_of_equity": ["us-gaap:PaymentsForRepurchaseOfCommonStock"],
                        "dividends_paid": ["us-gaap:PaymentsOfDividends", "us-gaap:DividendsPaid"]
                },
                "cash_flow_supplemental_and_other": {
                        "interest_paid_cf": ["us-gaap:InterestPaid"],
                        "income_taxes_paid_cf": ["us-gaap:IncomeTaxesPaid"],
                        "effect_of_exchange_rate_on_cash": ["us-gaap:EffectOfExchangeRateOnCashAndCashEquivalents"],
                        "cash_and_cash_equivalents_period_increase_decrease": ["us-gaap:CashAndCashEquivalentsPeriodIncreaseDecrease"]
                },
                "notes_share_based_compensation": {
                         "share_based_compensation_expense_note": ["us-gaap:ShareBasedCompensation", "us-gaap:StockBasedCompensation"],
                        "stock_options_outstanding_note": ["us-gaap:ShareBasedPaymentArrangementByShareBasedPaymentAwardOptionsOutstandingNumber"],
                        "restricted_stock_units_outstanding_note": ["us-gaap:ShareBasedPaymentArrangementByShareBasedPaymentAwardEquityInstrumentsOtherThanOptionsNonvestedNumber"]
                },
                "notes_debt_details": {
                        "long_term_debt_maturities_note": ["us-gaap:LongTermDebtMaturitiesRepaymentsOfPrincipalAfterYearFive"],
                        "debt_covenants_compliance_note": ["us-gaap:DebtInstrumentCovenantsCompliance"],
                        "weighted_average_interest_rate_on_debt_note": ["us-gaap:DebtWeightedAverageInterestRate"]
                },
                "notes_leases_details": {
                        "operating_lease_cost_note": ["us-gaap:LesseeOperatingLeaseExpense"],
                        "finance_lease_cost_note": ["us-gaap:LesseeFinanceLeaseCostInterestComponent", "us-gaap:LesseeFinanceLeaseCostAmortizationOfRightOfUseAsset"],
                        "future_minimum_lease_payments_operating_note": ["us-gaap:OperatingLeaseLiabilityFutureLeasePayments"],
                        "future_minimum_lease_payments_finance_note": ["us-gaap:FinanceLeaseLiabilityFutureLeasePayments"],
                        "weighted_average_lease_term_operating_note": ["us-gaap:OperatingLeaseWeightedAverageLeaseTerm"],
                        "weighted_average_discount_rate_operating_note": ["us-gaap:OperatingLeaseWeightedAverageDiscountRate"]
                },
                "notes_commitments_and_contingencies": {
                        "purchase_commitments_note": ["us-gaap:UnconditionalPurchaseObligationAmount"],
                        "legal_contingencies_accrual_note": ["us-gaap:AccrualForEnvironmentalLossContingencies"],
                        "guarantees_outstanding_note": ["us-gaap:GuaranteesOfIndebtednessOfOthersAmount"]
                },
                "notes_fair_value_measurements": {
                        "fair_value_assets_level_1_note": ["us-gaap:FairValueInputsLevel1Assets"],
                        "fair_value_assets_level_2_note": ["us-gaap:FairValueInputsLevel2Assets"],
                        "fair_value_assets_level_3_note": ["us-gaap:FairValueInputsLevel3Assets"],
                        "fair_value_liabilities_level_1_note": ["us-gaap:FairValueInputsLevel1Liabilities"],
                        "fair_value_liabilities_level_2_note": ["us-gaap:FairValueInputsLevel2Liabilities"],
                        "fair_value_liabilities_level_3_note": ["us-gaap:FairValueInputsLevel3Liabilities"]
                },
                "notes_derivatives_and_hedging": {
                        "notional_amount_of_interest_rate_swaps_note": ["us-gaap:InterestRateSwapNotionalAmount"],
                        "notional_amount_of_foreign_currency_contracts_note": ["us-gaap:ForeignCurrencyContractNotionalAmount"],
                        "gain_loss_on_derivatives_recognized_in_oci_note": ["us-gaap:AccumulatedOtherComprehensiveIncomeLossNetGainLossFromCashFlowHedgesEffectNetOfTax"],
                        "gain_loss_on_derivatives_recognized_in_income_note": ["us-gaap:DerivativeInstrumentGainLossReclassifiedFromAccumulatedOCIIntoIncomeEffectivePortionBeforeTax"]
                },
                "notes_income_taxes_details": {
                        "effective_income_tax_rate_reconciliation_note": ["us-gaap:EffectiveIncomeTaxRateReconciliationAtFederalStatutoryIncomeTaxRate"],
                        "deferred_tax_assets_by_type_note": ["us-gaap:DeferredTaxAssetsTaxCreditCarryforwards"],
                        "deferred_tax_liabilities_by_type_note": ["us-gaap:DeferredTaxLiabilitiesIntangibleAssets"],
                        "valuation_allowance_for_deferred_tax_assets_note": ["us-gaap:ValuationAllowanceOfDeferredTaxAssets"]
                },
                "notes_business_combinations_details": {
                        "purchase_price_allocation_goodwill_note": ["us-gaap:BusinessCombinationRecognizedIdentifiableAssetsAcquiredAndLiabilitiesAssumedGoodwill"],
                        "purchase_price_allocation_intangibles_note": ["us-gaap:BusinessCombinationRecognizedIdentifiableAssetsAcquiredAndLiabilitiesAssumedIntangibleAssetsOtherThanGoodwill"]
                },
                        "notes_related_party_transactions": {
                        "related_party_revenue_note": ["us-gaap:RevenueFromRelatedParties"],
                        "related_party_purchases_note": ["us-gaap:PurchasesFromRelatedParties"],
                        "due_from_related_parties_note": ["us-gaap:DueFromRelatedPartiesCurrent"],
                        "due_to_related_parties_note": ["us-gaap:DueToRelatedPartiesCurrent"]
                },
                "notes_segment_information": {
                        "segment_revenue_note": ["us-gaap:RevenueFromReportableSegments"],
                        "segment_profit_loss_note": ["us-gaap:SegmentReportingProfitLoss"],
                        "segment_assets_note": ["us-gaap:SegmentReportingAssets"]
                },
                "document_and_entity_information_dei": {
                        "document_type_dei": ["dei:DocumentType"],
                        "document_period_end_date_dei": ["dei:DocumentPeriodEndDate"],
                        "entity_registrant_name_dei": ["dei:EntityRegistrantName"],
                        "cik_dei": ["dei:EntityCentralIndexKey"],
                        "trading_symbol_dei": ["dei:TradingSymbol"],
                        "outstanding_shares_common_dei": ["dei:EntityCommonStockSharesOutstanding"],
                        "public_float_dei": ["dei:EntityPublicFloat"]
                },
                "other_disclosures_and_metrics": {
                        "research_and_development_assets_note": ["us-gaap:ResearchAndDevelopmentAssetNet"],
                        "environmental_liabilities_note": ["us-gaap:AccrualForEnvironmentalLossContingenciesNoncurrent"],
                        "restructuring_liability_note": ["us-gaap:RestructuringReserveCurrent"],
                        "off_balance_sheet_arrangements_note": ["us-gaap:OffBalanceSheetArrangement"],
                        "asset_retirement_obligation_note": ["us-gaap:AssetRetirementObligation"],
                        "employee_benefit_plans_note": ["us-gaap:DefinedBenefitPlanNetPeriodicBenefitCost"]
                        }
                },
            'require_submissions': True,
            'max_periods_to_analyze': 4,
            'include_amended_filings': True
        }
    
    def _default_email(self) -> Dict:
        """Default email configuration"""
        return {
            'enabled': False,
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'username': '',
            'password': '',
            'from_address': '',
            'recipients': [],
            'use_tls': True
        }
    
    def _default_analysis(self) -> Dict:
        """Default analysis configuration"""
        return {
            'fundamental_weight': 0.6,
            'technical_weight': 0.4,
            'min_score_for_buy': 7.0,
            'max_score_for_sell': 4.0,
            'lookback_days': 365,
            'min_volume': 100000
        }
    
    def _default_logging(self) -> Dict:
        """Default logging configuration"""
        return {
            'symbol_log_max_bytes': 1024 * 1024,  # 1MB
            'symbol_log_backup_count': 5,
            'symbol_log_format': '%(levelname)s - %(name)s - %(message)s',
            'main_log_max_bytes': 5 * 1024 * 1024,  # 5MB
            'main_log_backup_count': 10,
            'main_log_format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'main_log_file': 'investigator.log'
        }
    
    def _default_stocks(self) -> List[str]:
        """Default stock list"""
        return [
            'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA',
            'NVDA', 'META', 'NFLX', 'CRM', 'SNOW'
        ]
    
    def _default_parquet(self) -> Dict:
        """Default parquet configuration - uniform gzip compression"""
        return {
            'engine': 'fastparquet',
            'compression': 'gzip',
            'compression_level': 9,
            'pyarrow_compression': 'gzip',  # Uniform gzip
            'pyarrow_compression_level': 9,  # Maximum compression
            'use_dictionary': True,
            'row_group_size': None
        }
    
    def _default_cache_control(self) -> Dict:
        """Default cache control configuration"""
        return {
            'storage': ['disk', 'rdbms'],  # Both storage backends enabled by default
            'types': None,  # None means all cache types enabled
            'read_from_cache': True,
            'write_to_cache': True,
            'force_refresh': False,
            'force_refresh_symbols': None,  # List of symbols to force refresh
            'cache_ttl_override': None
        }
    
    def _build_caching_config(self, caching_data: Dict) -> CachingConfig:
        """Build caching configuration with user overrides"""
        caching_config = CachingConfig()
        
        # Apply user overrides for each cache type
        for cache_type, user_config in caching_data.items():
            if hasattr(caching_config, cache_type):
                current_config = getattr(caching_config, cache_type)
                
                # Update specific fields
                for field_name, value in user_config.items():
                    if field_name == 'storage_type' and isinstance(value, str):
                        value = CacheStorageType(value.lower())
                    setattr(current_config, field_name, value)
                    
                # Re-validate after updates
                current_config.__post_init__()
                
        return caching_config
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_file = os.path.join(self.logs_dir, 'investment_ai.log')
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def get_symbol_log_file(self, symbol: str) -> str:
        """Get symbol-specific log file path"""
        return str(self.logs_dir / f'{symbol}.log')
    
    def get_symbol_logger(self, symbol: str, module_name: str) -> logging.Logger:
        """Get symbol-specific logger with file rotation"""
        from logging.handlers import RotatingFileHandler
        
        logger_name = f"{module_name}.{symbol}"
        logger = logging.getLogger(logger_name)
        
        # Remove existing handlers to avoid duplicates
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Set logger level
        logger.setLevel(logging.INFO)
        logger.propagate = False  # Don't propagate to root logger
        
        # Create file handler with rotation
        log_file = self.get_symbol_log_file(symbol)
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=self.logging.symbol_log_max_bytes,
            backupCount=self.logging.symbol_log_backup_count
        )
        
        # Set formatter (no timestamp as requested)
        formatter = logging.Formatter(self.logging.symbol_log_format)
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(file_handler)
        
        return logger
    
    def get_main_logger(self, module_name: str = 'investigator') -> logging.Logger:
        """Get main consolidated logger with file rotation for non-symbol-specific messages"""
        from logging.handlers import RotatingFileHandler
        
        logger_name = f"main.{module_name}"
        logger = logging.getLogger(logger_name)
        
        # Remove existing handlers to avoid duplicates
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Set logger level
        logger.setLevel(logging.INFO)
        logger.propagate = False  # Don't propagate to root logger
        
        # Create file handler with rotation
        log_file = self.logs_dir / self.logging.main_log_file
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=self.logging.main_log_max_bytes,
            backupCount=self.logging.main_log_backup_count
        )
        
        # Set formatter with timestamp for main log
        formatter = logging.Formatter(self.logging.main_log_format)
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(file_handler)
        
        return logger
    
    def get_symbol_cache_path(self, symbol: str, cache_type: str = 'sec') -> Path:
        """Get symbol-specific cache directory path"""
        if cache_type == 'sec':
            # sec_cache/{symbol}/{category}_{formtype}_{period}.json
            cache_dir = Path(self.sec.cache_dir) / symbol
        elif cache_type == 'llm':
            # llm_cache/{symbol}/{category}_{formtype}_{period}.json  
            cache_dir = self.data_dir / 'llm_cache' / symbol
        else:
            cache_dir = self.data_dir / f'{cache_type}_cache' / symbol
        
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors"""
        errors = []
        
        # Validate SEC configuration
        if not self.sec.user_agent:
            errors.append("SEC user agent is required for API access")
        
        # Validate database connection details
        if not all([self.database.host, self.database.database, 
                   self.database.username, self.database.password]):
            errors.append("Database connection details are incomplete")
        
        # Validate email settings if enabled
        if self.email.enabled:
            if not all([self.email.username, self.email.password, 
                       self.email.smtp_server]):
                errors.append("Email settings are incomplete")
            
            if not self.email.recipients:
                errors.append("Email recipients list is empty")
        
        # Validate stock list
        if not self.stocks_to_track:
            errors.append("No stocks configured for tracking")
            
        # Validate caching configuration
        try:
            for cache_type in ['sec_submissions', 'sec_company_facts', 'sec_consolidated_frames',
                              'llm_sec_analysis', 'llm_technical_analysis', 'llm_synthesis_reports']:
                config = self.caching.get_cache_config(cache_type)
                if config and config.enabled:
                    if config.storage_type == CacheStorageType.DISK and not config.disk_path:
                        errors.append(f"Cache {cache_type}: disk_path required for disk storage")
                    elif config.storage_type == CacheStorageType.RDBMS and not config.table_name:
                        errors.append(f"Cache {cache_type}: table_name required for RDBMS storage")
        except Exception as e:
            errors.append(f"Caching configuration error: {e}")
        
        return errors
    
    def save_sample_config(self, filepath: str = "config.sample.json"):
        """Save a sample configuration file"""
        sample_config = {
            "database": {
                "host": "localhost",
                "port": 5432,
                "database": "investment_ai",
                "username": "investment_user", 
                "password": "investment_pass",
                "pool_size": 10,
                "max_overflow": 20
            },
            "ollama": {
                "base_url": "http://localhost:11434",
                "timeout": 600,
                "max_retries": 3,
                "min_context_size": 8192,
                "num_llm_threads": 1,
                "models": {
                    "fundamental_analysis": "mixtral-8x7b-q4_k_m-32K:latest",
                    "technical_analysis": "mixtral-8x7b-q4_k_m-32K:latest", 
                    "report_generation": "mixtral-8x7b-q4_k_m-32K:latest",
                    "synthesizer": "mixtral-8x7b-q4_k_m-32K:latest"
                },
                "num_predict": {
                    "fundamental_analysis": 2048,
                    "technical_analysis": 1536,
                    "report_generation": 2048,
                    "synthesizer": 1536
                }
            },
            "sec": {
                "user_agent": "InvestiGator/1.0 (singhvjd@gmail.com)",
                "base_url": "https://data.sec.gov",
                "rate_limit": 10,
                "cache_dir": "./data/sec_cache",
                "ticker_cache_file": "./data/ticker_cik_map.txt",
                "max_retries": 3,
                "timeout": 30,
                "require_submissions": true,
                "max_periods_to_analyze": 4,
                "include_amended_filings": true
            },
            "email": {
                "enabled": True,
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "username": "your-email@gmail.com",
                "password": "your-gmail-app-password",
                "from_address": "your-email@gmail.com",
                "recipients": ["your-email@gmail.com"],
                "use_tls": True
            },
            "analysis": {
                "fundamental_weight": 0.6,
                "technical_weight": 0.4,
                "min_score_for_buy": 7.0,
                "max_score_for_sell": 4.0,
                "lookback_days": 365,
                "min_volume": 100000
            },
            "logging": {
                "symbol_log_max_bytes": 1048576,
                "symbol_log_backup_count": 5,
                "symbol_log_format": "%(levelname)s - %(name)s - %(message)s",
                "main_log_max_bytes": 5242880,
                "main_log_backup_count": 10,
                "main_log_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "main_log_file": "investigator.log"
            },
            "stocks_to_track": [
                "NVDA", "META", "NFLX", "CRM", "SNOW", "FSLR", "AAPL"
            ],
            "data_dir": "./data",
            "reports_dir": "./reports", 
            "logs_dir": "./logs",
            "caching": {
                "sec_submissions": {
                    "enabled": True,
                    "storage_type": "disk",
                    "priority": 1,
                    "disk_path": "data/sec_cache",
                    "filename_pattern": "{symbol}_submissions.json",
                    "ttl_hours": 24
                },
                "sec_company_facts": {
                    "enabled": True,
                    "storage_type": "disk",
                    "priority": 1,
                    "disk_path": "data/sec_cache",
                    "filename_pattern": "{symbol}_companyfacts.json",
                    "ttl_hours": 24
                },
                "sec_consolidated_frames": {
                    "enabled": True,
                    "storage_type": "rdbms",
                    "priority": 1,
                    "table_name": "sec_consolidated_frames_cache",
                    "key_column": "cache_key",
                    "data_column": "consolidated_data",
                    "ttl_hours": 168
                },
                "llm_sec_analysis": {
                    "enabled": True,
                    "storage_type": "disk",
                    "priority": 1,
                    "disk_path": "data/llm_cache",
                    "filename_pattern": "{symbol}_sec_analysis_{period}.json",
                    "ttl_hours": 72
                },
                "llm_technical_analysis": {
                    "enabled": True,
                    "storage_type": "disk",
                    "priority": 1,
                    "disk_path": "data/llm_cache",
                    "filename_pattern": "{symbol}_technical_analysis.json",
                    "ttl_hours": 24
                },
                "llm_synthesis_reports": {
                    "enabled": True,
                    "storage_type": "rdbms",
                    "priority": 1,
                    "table_name": "llm_synthesis_cache",
                    "key_column": "report_key",
                    "data_column": "synthesis_data",
                    "ttl_hours": 168
                },
                "audit_sec_requests": {
                    "enabled": True,
                    "storage_type": "rdbms",
                    "priority": -1,
                    "table_name": "sec_api_requests_audit",
                    "key_column": "request_id",
                    "data_column": "request_data",
                    "max_entries": 10000
                },
                "audit_llm_interactions": {
                    "enabled": True,
                    "storage_type": "rdbms",
                    "priority": -1,
                    "table_name": "llm_interactions_audit",
                    "key_column": "interaction_id",
                    "data_column": "interaction_data",
                    "max_entries": 50000
                }
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(sample_config, f, indent=2)
        
        print(f"Sample configuration saved to {filepath}")

# Global configuration instance
_config_instance = None

def get_config() -> Config:
    """Get global configuration instance"""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance
