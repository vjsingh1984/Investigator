#!/usr/bin/env python3
"""
InvestiGator - Database Utilities Module
Copyright (c) 2025 Vijaykumar Singh
Licensed under the Apache License 2.0

Database Utilities Module
Handles all database operations using SQLAlchemy and PostgreSQL
"""

import logging
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

class StockAnalysis(Base):
    """Stock analysis table model"""
    __tablename__ = 'stock_analysis'
    
    symbol = Column(String(10), primary_key=True)
    fundamental_score = Column(Float)
    technical_score = Column(Float)
    overall_score = Column(Float)
    recommendation = Column(String(10))
    price_target = Column(Float)
    current_price = Column(Float)
    key_insights = Column(JSON)
    risks = Column(JSON)
    last_updated = Column(DateTime, default=datetime.utcnow)
    analysis_date = Column(DateTime, default=datetime.utcnow)
    full_analysis = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class SECFiling(Base):
    """SEC filings table model"""
    __tablename__ = 'sec_filings'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(10), nullable=False)
    form_type = Column(String(10), nullable=False)
    filing_date = Column(DateTime, nullable=False)
    period_end_date = Column(DateTime)
    filing_url = Column(Text, nullable=False)
    content_hash = Column(String(64))
    processed = Column(Boolean, default=False)
    filing_data = Column(JSON)
    fundamental_metrics = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

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

class AnalysisReport(Base):
    """Analysis reports table model"""
    __tablename__ = 'analysis_reports'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    report_date = Column(DateTime, nullable=False)
    report_type = Column(String(50), default='weekly')
    stocks_analyzed = Column(JSON)
    report_path = Column(Text)
    email_sent = Column(Boolean, default=False)
    email_sent_at = Column(DateTime)
    summary_stats = Column(JSON)
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

class StockAnalysisDAO:
    """Data Access Object for stock analysis operations"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
    
    def save_analysis(self, symbol: str, analysis_data: Dict) -> bool:
        """Save stock analysis to database"""
        try:
            with self.db.get_session() as session:
                # Check if analysis exists
                existing = session.query(StockAnalysis).filter_by(symbol=symbol).first()
                
                if existing:
                    # Update existing record
                    for key, value in analysis_data.items():
                        if hasattr(existing, key):
                            setattr(existing, key, value)
                    existing.updated_at = datetime.utcnow()
                else:
                    # Create new record
                    analysis = StockAnalysis(symbol=symbol, **analysis_data)
                    session.add(analysis)
                
                session.commit()
                logger.info(f"Saved analysis for {symbol}")
                return True
        except Exception as e:
            logger.error(f"Failed to save analysis for {symbol}: {e}")
            return False
    
    def get_analysis(self, symbol: str) -> Optional[Dict]:
        """Get latest analysis for a symbol"""
        try:
            with self.db.get_session() as session:
                analysis = session.query(StockAnalysis).filter_by(symbol=symbol).first()
                if analysis:
                    return {
                        'symbol': analysis.symbol,
                        'fundamental_score': analysis.fundamental_score,
                        'technical_score': analysis.technical_score,
                        'overall_score': analysis.overall_score,
                        'recommendation': analysis.recommendation,
                        'price_target': analysis.price_target,
                        'current_price': analysis.current_price,
                        'key_insights': analysis.key_insights,
                        'risks': analysis.risks,
                        'last_updated': analysis.last_updated,
                        'full_analysis': analysis.full_analysis
                    }
                return None
        except Exception as e:
            logger.error(f"Failed to get analysis for {symbol}: {e}")
            return None
    
    def get_all_analyses(self) -> List[Dict]:
        """Get all stock analyses"""
        try:
            with self.db.get_session() as session:
                analyses = session.query(StockAnalysis).all()
                return [
                    {
                        'symbol': a.symbol,
                        'fundamental_score': a.fundamental_score,
                        'technical_score': a.technical_score,
                        'overall_score': a.overall_score,
                        'recommendation': a.recommendation,
                        'price_target': a.price_target,
                        'current_price': a.current_price,
                        'key_insights': a.key_insights,
                        'risks': a.risks,
                        'last_updated': a.last_updated
                    }
                    for a in analyses
                ]
        except Exception as e:
            logger.error(f"Failed to get all analyses: {e}")
            return []

class SECFilingDAO:
    """Data Access Object for SEC filing operations"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
    
    def save_filing(self, filing_data: Dict) -> bool:
        """Save SEC filing to database"""
        try:
            with self.db.get_session() as session:
                filing = SECFiling(**filing_data)
                session.add(filing)
                session.commit()
                logger.info(f"Saved SEC filing for {filing_data.get('symbol')}")
                return True
        except Exception as e:
            logger.error(f"Failed to save SEC filing: {e}")
            return False
    
    def get_latest_filing(self, symbol: str, form_type: str = '10-K') -> Optional[Dict]:
        """Get latest filing for symbol and form type"""
        try:
            with self.db.get_session() as session:
                filing = session.query(SECFiling)\
                    .filter_by(symbol=symbol, form_type=form_type)\
                    .order_by(SECFiling.filing_date.desc())\
                    .first()
                
                if filing:
                    return {
                        'id': filing.id,
                        'symbol': filing.symbol,
                        'form_type': filing.form_type,
                        'filing_date': filing.filing_date,
                        'filing_url': filing.filing_url,
                        'filing_data': filing.filing_data,
                        'fundamental_metrics': filing.fundamental_metrics,
                        'processed': filing.processed
                    }
                return None
        except Exception as e:
            logger.error(f"Failed to get latest filing for {symbol}: {e}")
            return None

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

class ReportDAO:
    """Data Access Object for report operations"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
    
    def save_report(self, report_data: Dict) -> bool:
        """Save analysis report to database"""
        try:
            with self.db.get_session() as session:
                report = AnalysisReport(**report_data)
                session.add(report)
                session.commit()
                logger.info(f"Saved report: {report_data.get('report_type')}")
                return True
        except Exception as e:
            logger.error(f"Failed to save report: {e}")
            return False

class LLMResponseStoreDAO:
    """Data Access Object for LLM response store operations"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
    
    def save_llm_response(self, symbol: str, form_type: str, period: str,
                         prompt_context: Dict, model_info: Dict, response: Dict,
                         metadata: Dict, llm_type: str) -> bool:
        """Save LLM response to database"""
        try:
            with self.db.get_session() as session:
                session.execute(
                    text("""
                        INSERT INTO llm_response_store
                        (symbol, form_type, period, prompt_context, model_info, 
                         response, metadata, llm_type)
                        VALUES (:symbol, :form_type, :period, :prompt_context, 
                                :model_info, :response, :metadata, :llm_type)
                        ON CONFLICT (symbol, form_type, period, llm_type)
                        DO UPDATE SET
                            prompt_context = EXCLUDED.prompt_context,
                            model_info = EXCLUDED.model_info,
                            response = EXCLUDED.response,
                            metadata = EXCLUDED.metadata,
                            ts = NOW()
                    """),
                    {
                        'symbol': symbol,
                        'form_type': form_type,
                        'period': period,
                        'prompt_context': safe_json_dumps(prompt_context),
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
                query = "SELECT symbol, form_type, period, prompt_context, model_info, response, metadata, llm_type, ts FROM llm_response_store WHERE symbol = :symbol"
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
                        'prompt_context': result[3],
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
                    FROM llm_response_store
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
                        INSERT INTO sec_response_store
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
                        FROM sec_response_store
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
    """Data Access Object for all_submission_store operations"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
    
    def save_submission(self, symbol: str, cik: str, company_name: str, 
                       submissions_data: Dict, total_filings: Optional[int] = None,
                       latest_filing_date: Optional[str] = None, 
                       data_size_bytes: Optional[int] = None) -> bool:
        """Save or update submission data"""
        try:
            with self.db.get_session() as session:
                # Calculate data size if not provided
                if data_size_bytes is None:
                    data_size_bytes = len(safe_json_dumps(submissions_data).encode('utf-8'))
                
                # Extract total filings if not provided
                if total_filings is None and isinstance(submissions_data, dict):
                    filings = submissions_data.get('filings', {}).get('recent', {})
                    if filings and 'form' in filings:
                        total_filings = len(filings['form'])
                
                session.execute(
                    text("""
                        INSERT INTO all_submission_store 
                        (symbol, cik, company_name, submissions_data, total_filings, 
                         latest_filing_date, data_size_bytes, fetched_at, updated_at)
                        VALUES (:symbol, :cik, :company_name, :submissions_data, :total_filings,
                                :latest_filing_date, :data_size_bytes, NOW(), NOW())
                        ON CONFLICT (symbol, cik) DO UPDATE SET
                            company_name = EXCLUDED.company_name,
                            submissions_data = EXCLUDED.submissions_data,
                            total_filings = EXCLUDED.total_filings,
                            latest_filing_date = EXCLUDED.latest_filing_date,
                            data_size_bytes = EXCLUDED.data_size_bytes,
                            updated_at = NOW()
                    """),
                    {
                        'symbol': symbol,
                        'cik': cik,
                        'company_name': company_name,
                        'submissions_data': safe_json_dumps(submissions_data),
                        'total_filings': total_filings,
                        'latest_filing_date': latest_filing_date,
                        'data_size_bytes': data_size_bytes
                    }
                )
                session.commit()
                logger.info(f"Saved submissions for {symbol} (CIK: {cik})")
                return True
        except Exception as e:
            logger.error(f"Failed to save submission data: {e}")
            return False
    
    def get_submission(self, symbol: str, cik: str, max_age_days: int = 7) -> Optional[Dict]:
        """Get submission data if it exists and is recent enough"""
        try:
            with self.db.get_session() as session:
                result = session.execute(
                    text("""
                        SELECT submissions_data, company_name, total_filings, 
                               latest_filing_date, fetched_at, updated_at
                        FROM all_submission_store
                        WHERE symbol = :symbol AND cik = :cik
                        AND updated_at > NOW() - INTERVAL ':max_age days'
                        LIMIT 1
                    """.replace(':max_age', str(max_age_days))),
                    {'symbol': symbol, 'cik': cik}
                ).fetchone()
                
                if result:
                    return {
                        'submissions_data': result[0],
                        'company_name': result[1],
                        'total_filings': result[2],
                        'latest_filing_date': result[3],
                        'fetched_at': result[4],
                        'updated_at': result[5]
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
                        DELETE FROM all_submission_store
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
    
    def get_recent_earnings_submissions(self, symbol: str, form_types: List[str] = None, 
                                       limit: int = 4) -> List[Dict]:
        """Get recent earnings submissions from materialized view"""
        try:
            if form_types is None:
                form_types = ['10-K', '10-Q']
            
            with self.db.get_session() as session:
                # Build the query with form type filter
                form_type_clause = " AND form_type = ANY(:form_types)" if form_types else ""
                
                results = session.execute(
                    text(f"""
                        SELECT symbol, cik, company_name, form_type, filing_date,
                               accession_number, report_date, period, submission_info
                        FROM earnings_submission_materialized_view
                        WHERE symbol = :symbol{form_type_clause}
                        ORDER BY filing_date DESC
                        LIMIT :limit
                    """),
                    {
                        'symbol': symbol,
                        'form_types': form_types,
                        'limit': limit
                    }
                ).fetchall()
                
                submissions = []
                for row in results:
                    submissions.append({
                        'symbol': row[0],
                        'cik': row[1],
                        'company_name': row[2],
                        'form_type': row[3],
                        'filing_date': row[4],
                        'accession_number': row[5],
                        'report_date': row[6],
                        'period': row[7],
                        'submission_info': row[8]
                    })
                
                logger.info(f"Retrieved {len(submissions)} earnings submissions for {symbol}")
                return submissions
        except Exception as e:
            logger.error(f"Failed to get earnings submissions: {e}")
            return []
    
    def get_latest_10k_10q(self, symbol: str) -> Optional[Dict]:
        """Get the latest 10-K or 10-Q filing from materialized view"""
        try:
            with self.db.get_session() as session:
                result = session.execute(
                    text("""
                        SELECT symbol, cik, company_name, form_type, filing_date,
                               accession_number, report_date, period, submission_info
                        FROM earnings_submission_materialized_view
                        WHERE symbol = :symbol 
                        AND form_type IN ('10-K', '10-Q')
                        ORDER BY filing_date DESC
                        LIMIT 1
                    """),
                    {'symbol': symbol}
                ).fetchone()
                
                if result:
                    return {
                        'symbol': result[0],
                        'cik': result[1],
                        'company_name': result[2],
                        'form_type': result[3],
                        'filing_date': result[4],
                        'accession_number': result[5],
                        'report_date': result[6],
                        'period': result[7],
                        'submission_info': result[8]
                    }
                return None
        except Exception as e:
            logger.error(f"Failed to get latest 10-K/10-Q: {e}")
            return None
    
    def get_recent_earnings_submissions(self, symbol: str, limit: int = 4) -> List[Dict]:
        """Get recent earnings submissions from materialized view"""
        try:
            with self.db.get_session() as session:
                results = session.execute(
                    text("""
                        SELECT form_type, filing_date, accession_number, period, submission_info
                        FROM earnings_submission_materialized_view
                        WHERE symbol = :symbol
                        ORDER BY filing_date DESC
                        LIMIT :limit
                    """),
                    {'symbol': symbol, 'limit': limit}
                ).fetchall()
                
                submissions = []
                for result in results:
                    submissions.append({
                        'form_type': result[0],
                        'filing_date': result[1],
                        'accession_number': result[2],
                        'period': result[3],
                        'submission_info': result[4]
                    })
                
                return submissions
        except Exception as e:
            logger.error(f"Failed to get recent earnings submissions: {e}")
            return []

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

def get_stock_analysis_dao() -> StockAnalysisDAO:
    """Get stock analysis DAO"""
    return StockAnalysisDAO(get_db_manager())

def get_sec_filing_dao() -> SECFilingDAO:
    """Get SEC filing DAO"""
    return SECFilingDAO(get_db_manager())

def get_technical_indicators_dao() -> TechnicalIndicatorsDAO:
    """Get technical indicators DAO"""
    return TechnicalIndicatorsDAO(get_db_manager())

def get_report_dao() -> ReportDAO:
    """Get report DAO"""
    return ReportDAO(get_db_manager())

def get_llm_response_store_dao() -> LLMResponseStoreDAO:
    """Get LLM response store DAO instance"""
    return LLMResponseStoreDAO(get_db_manager())

def get_ticker_cik_mapping_dao() -> TickerCIKMappingDAO:
    """Get ticker-CIK mapping DAO instance"""
    return TickerCIKMappingDAO(get_db_manager())

def get_sec_response_store_dao() -> SECResponseStoreDAO:
    """Get SEC response store DAO instance"""
    return SECResponseStoreDAO(get_db_manager())

def get_quarterly_metrics_dao() -> QuarterlyMetricsDAO:
    """Get quarterly metrics DAO instance"""
    return QuarterlyMetricsDAO(get_db_manager())

def get_all_submission_store_dao() -> AllSubmissionStoreDAO:
    """Get all submission store DAO instance"""
    return AllSubmissionStoreDAO(get_db_manager())

class AllCompanyFactsStoreDAO:
    """DAO for managing Company Facts data using cache manager pattern with compression"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    
    def store_company_facts(self, symbol: str, cik: str, company_name: str, 
                          companyfacts: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Store company facts data for a symbol"""
        try:
            with self.db_manager.get_session() as session:
                # Check if record exists
                existing = session.execute(
                    text("SELECT symbol FROM all_companyfacts_store WHERE symbol = :symbol"),
                    {"symbol": symbol}
                ).fetchone()
                
                if existing:
                    # Update existing record - validate CIK first
                    cik_int = int(cik) if cik and str(cik).strip() else None
                    if not cik_int or cik_int <= 0:
                        logger.error(f"Invalid CIK for {symbol}: {cik}")
                        return False
                    
                    session.execute(
                        text("""
                            UPDATE all_companyfacts_store 
                            SET cik = :cik, company_name = :company_name, 
                                companyfacts = :companyfacts, metadata = :metadata,
                                updated_at = :updated_at
                            WHERE symbol = :symbol
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
                    logger.debug(f"Updated company facts for {symbol}")
                else:
                    # Insert new record - validate CIK first
                    cik_int = int(cik) if cik and str(cik).strip() else None
                    if not cik_int or cik_int <= 0:
                        logger.error(f"Invalid CIK for {symbol}: {cik}")
                        return False
                    
                    session.execute(
                        text("""
                            INSERT INTO all_companyfacts_store 
                            (symbol, cik, company_name, companyfacts, metadata)
                            VALUES (:symbol, :cik, :company_name, :companyfacts, :metadata)
                        """),
                        {
                            "symbol": symbol,
                            "cik": cik_int,
                            "company_name": company_name,
                            "companyfacts": safe_json_dumps(companyfacts),
                            "metadata": safe_json_dumps(metadata) if metadata else None
                        }
                    )
                    logger.debug(f"Inserted company facts for {symbol}")
                
                session.commit()
                return True
                
        except Exception as e:
            logger.error(f"Error storing company facts for {symbol}: {e}")
            return False
    
    def get_company_facts(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Retrieve company facts for a symbol"""
        try:
            with self.db_manager.get_session() as session:
                result = session.execute(
                    text("""
                        SELECT companyfacts, metadata, cik, company_name, updated_at
                        FROM all_companyfacts_store 
                        WHERE symbol = :symbol
                    """),
                    {"symbol": symbol}
                ).fetchone()
                
                if result:
                    companyfacts, metadata, cik, company_name, updated_at = result
                    return {
                        "companyfacts": safe_json_loads(companyfacts),
                        "metadata": safe_json_loads(metadata) if metadata else {},
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
                    text("SELECT symbol FROM all_companyfacts_store ORDER BY symbol")
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
                        DELETE FROM all_companyfacts_store 
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

def get_all_companyfacts_store_dao() -> AllCompanyFactsStoreDAO:
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
