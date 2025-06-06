#!/usr/bin/env python3
"""
InvestiGator - Database Utilities Module
Copyright (c) 2025 Vijaykumar Singh
Licensed under the Apache License 2.0

Database Utilities Module
Handles all database operations using SQLAlchemy and PostgreSQL
"""

import logging
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple
from contextlib import contextmanager
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, String, Float, DateTime, Text, Integer, Boolean, JSON
from sqlalchemy.dialects.postgresql import UUID
import json
import uuid
import io

from config import get_config

# UTF-8 encoding helpers for JSON operations
def safe_json_dumps(obj: Any, **kwargs) -> str:
    """Safely encode object to JSON with UTF-8 encoding, handling binary characters"""
    return json.dumps(obj, ensure_ascii=False, **kwargs)

def safe_json_loads(json_str: str) -> Any:
    """Safely decode JSON string with UTF-8 encoding"""
    if isinstance(json_str, bytes):
        json_str = json_str.decode('utf-8', errors='replace')
    return json.loads(json_str)

logger = logging.getLogger(__name__)

# SQLAlchemy Base
Base = declarative_base()

class TechnicalIndicators(Base):
    """Technical indicators table model"""
    __tablename__ = 'technical_indicators'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(10), nullable=False)
    analysis_date = Column(DateTime, nullable=False)
    current_price = Column(Float)
    sma_20 = Column(Float)
    sma_50 = Column(Float)
    sma_200 = Column(Float)
    rsi = Column(Float)
    macd = Column(Float)
    macd_signal = Column(Float)
    volume = Column(Float)
    price_change_1d = Column(Float)
    price_change_1w = Column(Float)
    price_change_1m = Column(Float)
    technical_score = Column(Float)
    indicators_data = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

class DatabaseManager:
    """Database manager class"""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.engine = None
        self.SessionLocal = None
        self._initialize_engine()
    
    def _initialize_engine(self):
        """Initialize database engine and session factory"""
        try:
            self.engine = create_engine(
                self.config.database.url,
                pool_size=self.config.database.pool_size,
                max_overflow=self.config.database.max_overflow,
                echo=False,  # Set to True for SQL debugging
                pool_pre_ping=True
            )
            
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )
            
            logger.info("Database engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database engine: {e}")
            raise
    
    @contextmanager
    def get_session(self):
        """Get database session context manager"""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            import traceback
            logger.error(f"Database session error: {e}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            raise
        finally:
            session.close()
    
    def create_tables(self):
        """Create all database tables"""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create database tables: {e}")
            raise
    
    def execute_sql_file(self, sql_file_path: str):
        """Execute SQL commands from file"""
        try:
            with open(sql_file_path, 'r') as f:
                sql_content = f.read()
            
            with self.engine.begin() as conn:
                # Split by semicolon and execute each statement
                statements = [stmt.strip() for stmt in sql_content.split(';') if stmt.strip()]
                for statement in statements:
                    if statement:
                        conn.execute(text(statement))
            
            logger.info(f"SQL file {sql_file_path} executed successfully")
        except Exception as e:
            logger.error(f"Failed to execute SQL file {sql_file_path}: {e}")
            raise
    
    def test_connection(self) -> bool:
        """Test database connection"""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Database connection test successful")
            return True
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False

class TechnicalIndicatorsDAO:
    """Data Access Object for technical indicators operations"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
    
    def save_indicators(self, indicators_data: Dict) -> bool:
        """Save technical indicators to database"""
        try:
            with self.db.get_session() as session:
                indicators = TechnicalIndicators(**indicators_data)
                session.add(indicators)
                session.commit()
                logger.info(f"Saved technical indicators for {indicators_data.get('symbol')}")
                return True
        except Exception as e:
            logger.error(f"Failed to save technical indicators: {e}")
            return False
    
    def get_latest_indicators(self, symbol: str) -> Optional[Dict]:
        """Get latest technical indicators for symbol"""
        try:
            with self.db.get_session() as session:
                indicators = session.query(TechnicalIndicators)\
                    .filter_by(symbol=symbol)\
                    .order_by(TechnicalIndicators.analysis_date.desc())\
                    .first()
                
                if indicators:
                    return {
                        'symbol': indicators.symbol,
                        'analysis_date': indicators.analysis_date,
                        'current_price': indicators.current_price,
                        'sma_20': indicators.sma_20,
                        'sma_50': indicators.sma_50,
                        'sma_200': indicators.sma_200,
                        'rsi': indicators.rsi,
                        'macd': indicators.macd,
                        'macd_signal': indicators.macd_signal,
                        'volume': indicators.volume,
                        'price_change_1d': indicators.price_change_1d,
                        'price_change_1w': indicators.price_change_1w,
                        'price_change_1m': indicators.price_change_1m,
                        'technical_score': indicators.technical_score,
                        'indicators_data': indicators.indicators_data
                    }
                return None
        except Exception as e:
            logger.error(f"Failed to get technical indicators for {symbol}: {e}")
            return None

class LLMResponseStoreDAO:
    """Data Access Object for LLM response store operations"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
    
    def save_llm_response(self, symbol: str, form_type: str, period: str,
                         prompt: str, model_info: Dict, response: Dict,
                         metadata: Dict, llm_type: str) -> bool:
        """Save LLM response to database"""
        try:
            with self.db.get_session() as session:
                session.execute(
                    text("""
                        INSERT INTO llm_responses
                        (symbol, form_type, period, prompt, model_info, 
                         response, metadata, llm_type)
                        VALUES (:symbol, :form_type, :period, :prompt, 
                                :model_info, :response, :metadata, :llm_type)
                        ON CONFLICT (symbol, form_type, period, llm_type)
                        DO UPDATE SET
                            prompt = EXCLUDED.prompt,
                            model_info = EXCLUDED.model_info,
                            response = EXCLUDED.response,
                            metadata = EXCLUDED.metadata,
                            ts = NOW()
                    """),
                    {
                        'symbol': symbol,
                        'form_type': form_type,
                        'period': period,
                        'prompt': prompt,
                        'model_info': safe_json_dumps(model_info),
                        'response': safe_json_dumps(response),
                        'metadata': safe_json_dumps(metadata),
                        'llm_type': llm_type
                    }
                )
                session.commit()
                logger.info(f"Saved LLM response for {symbol} {period} {llm_type}")
                return True
        except Exception as e:
            logger.error(f"Failed to save LLM response: {e}")
            return False
    
    def get_llm_response(self, symbol: str, form_type: str = None, 
                        period: str = None, llm_type: str = None) -> Optional[Dict]:
        """Get LLM response from database"""
        try:
            with self.db.get_session() as session:
                query = "SELECT symbol, form_type, period, prompt, model_info, response, metadata, llm_type, ts FROM llm_responses WHERE symbol = :symbol"
                params = {'symbol': symbol}
                
                if form_type is not None:
                    query += " AND form_type = :form_type"
                    params['form_type'] = form_type
                if period is not None:
                    query += " AND period = :period"
                    params['period'] = period
                if llm_type is not None:
                    query += " AND llm_type = :llm_type"
                    params['llm_type'] = llm_type
                
                query += " ORDER BY ts DESC LIMIT 1"
                
                result = session.execute(text(query), params).fetchone()
                
                if result:
                    return {
                        'symbol': result[0],
                        'form_type': result[1],
                        'period': result[2],
                        'prompt': result[3],
                        'model_info': result[4],
                        'response': result[5],
                        'metadata': result[6],
                        'llm_type': result[7],
                        'ts': result[8]
                    }
                return None
        except Exception as e:
            logger.error(f"Failed to get LLM response: {e}")
            return None
    
    def get_llm_responses_by_symbol(self, symbol: str, llm_type: str = None) -> List[Dict]:
        """Get all LLM responses for a symbol"""
        try:
            with self.db.get_session() as session:
                query = """
                    SELECT symbol, form_type, period, llm_type, 
                           metadata->>'processing_time_ms' as processing_time,
                           metadata->>'response_length' as response_length,
                           ts
                    FROM llm_responses
                    WHERE symbol = :symbol
                """
                params = {'symbol': symbol}
                
                if llm_type:
                    query += " AND llm_type = :llm_type"
                    params['llm_type'] = llm_type
                
                query += " ORDER BY ts DESC"
                
                results = session.execute(text(query), params).fetchall()
                
                return [
                    {
                        'symbol': r[0],
                        'form_type': r[1],
                        'period': r[2],
                        'llm_type': r[3],
                        'processing_time_ms': r[4],
                        'response_length': r[5],
                        'ts': r[6]
                    }
                    for r in results
                ]
        except Exception as e:
            logger.error(f"Failed to get LLM responses: {e}")
            return []
    
    def delete_llm_responses(self, symbol: str, form_type: str = None, 
                            period: str = None, llm_type: str = None) -> int:
        """Delete LLM responses matching criteria"""
        try:
            with self.db.get_session() as session:
                query = "DELETE FROM llm_responses WHERE symbol = :symbol"
                params = {'symbol': symbol}
                
                if form_type is not None:
                    query += " AND form_type = :form_type"
                    params['form_type'] = form_type
                if period is not None:
                    query += " AND period = :period"
                    params['period'] = period
                if llm_type is not None:
                    query += " AND llm_type = :llm_type"
                    params['llm_type'] = llm_type
                
                result = session.execute(text(query), params)
                session.commit()
                
                deleted_count = result.rowcount
                logger.info(f"Deleted {deleted_count} LLM responses for {symbol}")
                return deleted_count
        except Exception as e:
            logger.error(f"Failed to delete LLM responses: {e}")
            return 0
    
    def delete_llm_responses_by_pattern(self, symbol_pattern: str = None, 
                                       form_type_pattern: str = None) -> int:
        """Delete LLM responses matching patterns"""
        try:
            with self.db.get_session() as session:
                query = "DELETE FROM llm_responses WHERE 1=1"
                params = {}
                
                if symbol_pattern is not None:
                    query += " AND symbol LIKE :symbol_pattern"
                    params['symbol_pattern'] = symbol_pattern
                if form_type_pattern is not None:
                    query += " AND form_type LIKE :form_type_pattern"
                    params['form_type_pattern'] = form_type_pattern
                
                result = session.execute(text(query), params)
                session.commit()
                
                deleted_count = result.rowcount
                logger.info(f"Deleted {deleted_count} LLM responses by pattern")
                return deleted_count
        except Exception as e:
            logger.error(f"Failed to delete LLM responses by pattern: {e}")
            return 0

class TickerCIKMappingDAO:
    """Data Access Object for ticker-CIK mapping operations"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
    
    def save_mapping(self, ticker: str, cik: str, company_name: str, 
                    exchange: str = None) -> bool:
        """Save ticker-CIK mapping"""
        try:
            with self.db.get_session() as session:
                session.execute(
                    text("""
                        INSERT INTO ticker_cik_mapping (ticker, cik, company_name, exchange)
                        VALUES (:ticker, :cik, :company_name, :exchange)
                        ON CONFLICT (ticker) DO UPDATE SET
                            cik = EXCLUDED.cik,
                            company_name = EXCLUDED.company_name,
                            exchange = EXCLUDED.exchange,
                            updated_at = NOW()
                    """),
                    {
                        'ticker': ticker,
                        'cik': cik,
                        'company_name': company_name,
                        'exchange': exchange
                    }
                )
                session.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to save ticker mapping: {e}")
            return False
    
    def get_cik(self, ticker: str) -> Optional[str]:
        """Get CIK for ticker"""
        try:
            with self.db.get_session() as session:
                result = session.execute(
                    text("SELECT cik FROM ticker_cik_mapping WHERE ticker = :ticker"),
                    {'ticker': ticker}
                ).fetchone()
                return result[0] if result else None
        except Exception as e:
            logger.error(f"Failed to get CIK for {ticker}: {e}")
            return None
    
    def get_all_mappings(self) -> List[Dict]:
        """Get all ticker-CIK mappings"""
        try:
            with self.db.get_session() as session:
                results = session.execute(
                    text("SELECT ticker, cik, company_name, exchange FROM ticker_cik_mapping")
                ).fetchall()
                
                return [
                    {
                        'ticker': r[0],
                        'cik': r[1],
                        'company_name': r[2],
                        'exchange': r[3]
                    }
                    for r in results
                ]
        except Exception as e:
            logger.error(f"Failed to get all mappings: {e}")
            return []

class SECResponseStoreDAO:
    """Data Access Object for SEC response store operations"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
    
    def save_response(self, symbol: str, category: str, period: str,
                     response: Dict, metadata: Dict) -> bool:
        """Save SEC response data"""
        try:
            with self.db.get_session() as session:
                # Extract form_type from metadata or default to '10-K'
                form_type = metadata.get('form_type', '10-K')
                api_url = metadata.get('api_url', '')
                
                session.execute(
                    text("""
                        INSERT INTO sec_responses
                        (symbol, form_type, period, category, api_url, response, metadata)
                        VALUES (:symbol, :form_type, :period, :category, :api_url, :response, :metadata)
                        ON CONFLICT (symbol, form_type, period, category) DO UPDATE SET
                            api_url = EXCLUDED.api_url,
                            response = EXCLUDED.response,
                            metadata = EXCLUDED.metadata,
                            ts = NOW()
                    """),
                    {
                        'symbol': symbol,
                        'form_type': form_type,
                        'category': category,
                        'period': period,
                        'api_url': api_url,
                        'response': safe_json_dumps(response),
                        'metadata': safe_json_dumps(metadata)
                    }
                )
                session.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to save SEC response: {e}")
            return False
    
    def get_response(self, symbol: str, category: str, period: str) -> Optional[Dict]:
        """Get SEC response data"""
        try:
            with self.db.get_session() as session:
                result = session.execute(
                    text("""
                        SELECT response, metadata, ts
                        FROM sec_responses
                        WHERE symbol = :symbol 
                          AND category = :category
                          AND period = :period
                          AND ts > NOW() - INTERVAL '7 days'
                    """),
                    {
                        'symbol': symbol,
                        'category': category,
                        'period': period
                    }
                ).fetchone()
                
                if result:
                    return {
                        'response': result[0],
                        'metadata': result[1],
                        'ts': result[2]
                    }
                return None
        except Exception as e:
            logger.error(f"Failed to get SEC response: {e}")
            return None

class AllSubmissionStoreDAO:
    """Data Access Object for sec_submissions operations"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
    
    def save_submission(self, symbol: str, cik: str, company_name: str, 
                       submissions_data: Dict) -> bool:
        """Save or update submission data - simple storage only"""
        try:
            with self.db.get_session() as session:
                session.execute(
                    text("""
                        INSERT INTO sec_submissions 
                        (symbol, cik, company_name, submissions_data, fetched_at, updated_at)
                        VALUES (:symbol, :cik, :company_name, :submissions_data, NOW(), NOW())
                        ON CONFLICT (symbol, cik) DO UPDATE SET
                            company_name = EXCLUDED.company_name,
                            submissions_data = EXCLUDED.submissions_data,
                            updated_at = NOW()
                    """),
                    {
                        'symbol': symbol,
                        'cik': cik,
                        'company_name': company_name,
                        'submissions_data': safe_json_dumps(submissions_data)
                    }
                )
                session.commit()
                logger.info(f"Saved submissions for {symbol} (CIK: {cik})")
                return True
        except Exception as e:
            logger.error(f"Failed to save submission data: {e}")
            return False
    
    def get_submission(self, symbol: str, cik: str, max_age_days: int = 7) -> Optional[Dict]:
        """Get submission data if it exists and is recent enough - simple retrieval only"""
        try:
            with self.db.get_session() as session:
                result = session.execute(
                    text("""
                        SELECT submissions_data, company_name, fetched_at, updated_at
                        FROM sec_submissions
                        WHERE symbol = :symbol AND cik = :cik
                        AND updated_at > NOW() - INTERVAL ':max_age days'
                        LIMIT 1
                    """.replace(':max_age', str(max_age_days))),
                    {'symbol': symbol, 'cik': cik}
                ).fetchone()
                
                if result:
                    # Return raw data - let Python code process it
                    submissions_data = json.loads(result[0]) if isinstance(result[0], str) else result[0]
                    return {
                        'submissions_data': submissions_data,
                        'company_name': result[1],
                        'fetched_at': result[2],
                        'updated_at': result[3]
                    }
                return None
        except Exception as e:
            logger.error(f"Failed to get submission data: {e}")
            return None
    
    def delete_old_submissions(self, days_to_keep: int = 30) -> int:
        """Delete submissions older than specified days"""
        try:
            with self.db.get_session() as session:
                result = session.execute(
                    text("""
                        DELETE FROM sec_submissions
                        WHERE updated_at < NOW() - INTERVAL ':days days'
                    """.replace(':days', str(days_to_keep)))
                )
                session.commit()
                deleted_count = result.rowcount
                logger.info(f"Deleted {deleted_count} old submission records")
                return deleted_count
        except Exception as e:
            logger.error(f"Failed to delete old submissions: {e}")
            return 0
    
    # NOTE: get_recent_earnings_submissions method removed
    # Submissions now come from cache manager interface
    
    # NOTE: get_latest_10k_10q method removed
    # Latest filings now come from cache manager interface
    
    # NOTE: Second get_recent_earnings_submissions method also removed
    # All submission data now comes from cache handlers, not materialized views

class QuarterlyMetricsDAO:
    """Data Access Object for quarterly metrics operations"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
    
    def save_metrics(self, cik: str, ticker: str, fiscal_year: int,
                    fiscal_period: str, form_type: str, category: str,
                    concept_data: Dict, common_metadata: Dict = None) -> bool:
        """Save quarterly metrics"""
        try:
            with self.db.get_session() as session:
                session.execute(
                    text("""
                        INSERT INTO quarterly_metrics
                        (cik, ticker, fiscal_year, fiscal_period, form_type, 
                         category, concept_data, common_metadata)
                        VALUES (:cik, :ticker, :fiscal_year, :fiscal_period, 
                                :form_type, :category, :concept_data, :common_metadata)
                    """),
                    {
                        'cik': cik,
                        'ticker': ticker,
                        'fiscal_year': fiscal_year,
                        'fiscal_period': fiscal_period,
                        'form_type': form_type,
                        'category': category,
                        'concept_data': safe_json_dumps(concept_data),
                        'common_metadata': safe_json_dumps(common_metadata) if common_metadata else None
                    }
                )
                session.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to save quarterly metrics: {e}")
            return False
    
    def get_metrics(self, ticker: str, fiscal_year: int = None, 
                   fiscal_period: str = None) -> List[Dict]:
        """Get quarterly metrics for ticker"""
        try:
            with self.db.get_session() as session:
                query = """
                    SELECT fiscal_year, fiscal_period, form_type, category, 
                           concept_data, common_metadata, created_at
                    FROM quarterly_metrics
                    WHERE ticker = :ticker
                """
                params = {'ticker': ticker}
                
                if fiscal_year:
                    query += " AND fiscal_year = :fiscal_year"
                    params['fiscal_year'] = fiscal_year
                if fiscal_period:
                    query += " AND fiscal_period = :fiscal_period"
                    params['fiscal_period'] = fiscal_period
                
                query += " ORDER BY fiscal_year DESC, fiscal_period"
                
                results = session.execute(text(query), params).fetchall()
                
                return [
                    {
                        'fiscal_year': r[0],
                        'fiscal_period': r[1],
                        'form_type': r[2],
                        'category': r[3],
                        'concept_data': r[4],
                        'common_metadata': r[5],
                        'created_at': r[6]
                    }
                    for r in results
                ]
        except Exception as e:
            logger.error(f"Failed to get quarterly metrics: {e}")
            return []

# Global database manager instance
_db_manager = None

def get_db_manager() -> DatabaseManager:
    """Get global database manager instance"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager

def get_technical_indicators_dao() -> TechnicalIndicatorsDAO:
    """Get technical indicators DAO"""
    return TechnicalIndicatorsDAO(get_db_manager())

def get_llm_responses_dao() -> LLMResponseStoreDAO:
    """Get LLM response store DAO instance"""
    return LLMResponseStoreDAO(get_db_manager())

def get_ticker_cik_mapping_dao() -> TickerCIKMappingDAO:
    """Get ticker-CIK mapping DAO instance"""
    return TickerCIKMappingDAO(get_db_manager())

def get_sec_responses_dao() -> SECResponseStoreDAO:
    """Get SEC response store DAO instance"""
    return SECResponseStoreDAO(get_db_manager())

def get_quarterly_metrics_dao() -> QuarterlyMetricsDAO:
    """Get quarterly metrics DAO instance"""
    return QuarterlyMetricsDAO(get_db_manager())

def get_sec_submissions_dao() -> AllSubmissionStoreDAO:
    """Get all submission store DAO instance"""
    return AllSubmissionStoreDAO(get_db_manager())

class AllCompanyFactsStoreDAO:
    """DAO for managing Company Facts data using cache manager pattern with compression"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    
    def store_company_facts(self, symbol: str, cik: str, company_name: str, 
                          companyfacts: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Store company facts data for a symbol using optimized upsert"""
        try:
            # Validate CIK first
            cik_int = int(cik) if cik and str(cik).strip() else None
            if not cik_int or cik_int <= 0:
                logger.error(f"Invalid CIK for {symbol}: {cik}")
                return False
            
            with self.db_manager.get_session() as session:
                # Use PostgreSQL UPSERT (INSERT ... ON CONFLICT DO UPDATE)
                session.execute(
                    text("""
                        INSERT INTO sec_companyfacts 
                        (symbol, cik, company_name, companyfacts, metadata, updated_at)
                        VALUES (:symbol, :cik, :company_name, :companyfacts, :metadata, :updated_at)
                        ON CONFLICT (symbol, cik) DO UPDATE SET
                            company_name = EXCLUDED.company_name,
                            companyfacts = EXCLUDED.companyfacts,
                            metadata = EXCLUDED.metadata,
                            updated_at = EXCLUDED.updated_at
                    """),
                    {
                        "symbol": symbol,
                        "cik": cik_int,
                        "company_name": company_name,
                        "companyfacts": safe_json_dumps(companyfacts),
                        "metadata": safe_json_dumps(metadata) if metadata else None,
                        "updated_at": datetime.utcnow()
                    }
                )
                session.commit()
                logger.debug(f"Upserted company facts for {symbol}")
                return True
                
        except Exception as e:
            logger.error(f"Error storing company facts for {symbol}: {e}")
            return False
    
    def get_company_facts(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Retrieve company facts for a symbol"""
        try:
            logger.info(f"🔍 DB GET_COMPANY_FACTS: Starting query for {symbol}")
            with self.db_manager.get_session() as session:
                logger.info(f"🔍 DB GET_COMPANY_FACTS: Got session for {symbol}")
                result = session.execute(
                    text("""
                        SELECT companyfacts, metadata, cik, company_name, updated_at
                        FROM sec_companyfacts 
                        WHERE symbol = :symbol
                    """),
                    {"symbol": symbol}
                ).fetchone()
                logger.info(f"🔍 DB GET_COMPANY_FACTS: Query executed for {symbol}, result: {result is not None}")
                
                if result:
                    logger.info(f"🔍 DB GET_COMPANY_FACTS: Processing result for {symbol}")
                    companyfacts, metadata, cik, company_name, updated_at = result
                    logger.info(f"🔍 DB GET_COMPANY_FACTS: Data extracted, companyfacts type: {type(companyfacts)}")
                    
                    # Process companyfacts
                    if isinstance(companyfacts, dict):
                        logger.info(f"🔍 DB GET_COMPANY_FACTS: companyfacts is dict for {symbol}")
                        processed_facts = companyfacts
                    else:
                        logger.info(f"🔍 DB GET_COMPANY_FACTS: parsing companyfacts string for {symbol}")
                        processed_facts = safe_json_loads(companyfacts)
                        logger.info(f"🔍 DB GET_COMPANY_FACTS: companyfacts parsed for {symbol}")
                    
                    # Process metadata
                    if metadata:
                        if isinstance(metadata, dict):
                            processed_metadata = metadata
                        else:
                            processed_metadata = safe_json_loads(metadata)
                    else:
                        processed_metadata = {}
                    
                    logger.info(f"🔍 DB GET_COMPANY_FACTS: Building return dict for {symbol}")
                    return {
                        "companyfacts": processed_facts,
                        "metadata": processed_metadata,
                        "symbol": symbol,
                        "cik": cik,
                        "company_name": company_name,
                        "updated_at": updated_at
                    }
                    
        except Exception as e:
            logger.error(f"Error retrieving company facts for {symbol}: {e}")
        
        return None
    
    def get_all_symbols(self) -> List[str]:
        """Get all symbols that have company facts stored"""
        try:
            with self.db_manager.get_session() as session:
                results = session.execute(
                    text("SELECT symbol FROM sec_companyfacts ORDER BY symbol")
                ).fetchall()
                
                return [row[0] for row in results]
                
        except Exception as e:
            logger.error(f"Error retrieving all symbols: {e}")
            return []
    
    def cleanup_old_data(self, days_to_keep: int = 7) -> int:
        """Clean up old company facts data"""
        try:
            with self.db_manager.get_session() as session:
                result = session.execute(
                    text("""
                        DELETE FROM sec_companyfacts 
                        WHERE updated_at < :cutoff_date
                    """),
                    {"cutoff_date": datetime.utcnow() - timedelta(days=days_to_keep)}
                )
                session.commit()
                
                deleted_count = result.rowcount
                logger.info(f"Cleaned up {deleted_count} old company facts records")
                return deleted_count
                
        except Exception as e:
            logger.error(f"Error cleaning up old company facts data: {e}")
            return 0

def get_sec_companyfacts_dao() -> AllCompanyFactsStoreDAO:
    """Get Company Facts store DAO instance"""
    return AllCompanyFactsStoreDAO(get_db_manager())

if __name__ == "__main__":
    # Test database connection
    db_manager = get_db_manager()
    
    if db_manager.test_connection():
        print("✅ Database connection successful")
        db_manager.create_tables()
        print("✅ Database tables created")
    else:
        print("❌ Database connection failed")
