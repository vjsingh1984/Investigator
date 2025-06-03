-- InvestiGator Consolidated Database Schema
-- This file contains ALL active database schema definitions
-- Copyright (c) 2025 Vijaykumar Singh
-- Licensed under the Apache License, Version 2.0
-- Created: 2025-01-30
-- Last Updated: 2025-01-30
-- Version: 3.0.0 (Consolidated)

-- ================================================================================================
-- INITIAL SETUP
-- ================================================================================================

-- Create database if not exists (run manually first)
-- CREATE DATABASE investment_ai;
-- CREATE USER investment_user WITH PASSWORD 'investment_pass';
-- GRANT ALL PRIVILEGES ON DATABASE investment_ai TO investment_user;

-- Extension for UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ================================================================================================
-- CORE TABLES (ACTIVE)
-- ================================================================================================

-- Ticker to CIK mapping table
CREATE TABLE IF NOT EXISTS ticker_cik_mapping (
    ticker VARCHAR(10) PRIMARY KEY,
    cik VARCHAR(10) NOT NULL,
    company_name VARCHAR(255),
    exchange VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Enhanced SEC submissions table (v2)
CREATE TABLE IF NOT EXISTS sec_submissions_v2 (
    -- Primary identification
    id SERIAL PRIMARY KEY,
    cik VARCHAR(10) NOT NULL,
    ticker VARCHAR(10),
    company_name VARCHAR(255),
    
    -- Submission metadata
    entity_type VARCHAR(50),
    sic VARCHAR(4),
    sic_description VARCHAR(255),
    fiscal_year_end VARCHAR(4),
    state_of_incorporation VARCHAR(2),
    
    -- Company information
    website TEXT,
    investor_website TEXT,
    category VARCHAR(50),
    description TEXT,
    phone VARCHAR(50),
    
    -- Parsed submission data (JSONB with compression)
    parsed_data JSONB NOT NULL,  -- Output from submission_processor.parse_submissions()
    
    -- Raw SEC data (compressed)
    raw_data JSONB,  -- Original SEC API response
    
    -- Caching metadata
    data_source VARCHAR(50) DEFAULT 'sec_edgar_api',
    cache_version INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT uk_sec_submissions_v2_cik UNIQUE(cik)
);

-- Individual filings table for better querying
CREATE TABLE IF NOT EXISTS sec_filings (
    id SERIAL PRIMARY KEY,
    cik VARCHAR(10) NOT NULL,
    ticker VARCHAR(10),
    
    -- Filing details
    form_type VARCHAR(20) NOT NULL,
    filing_date DATE NOT NULL,
    accession_number VARCHAR(25) NOT NULL,
    primary_document VARCHAR(255),
    report_date DATE,
    
    -- Fiscal period information
    fiscal_year INTEGER,
    fiscal_period VARCHAR(10),  -- Q1, Q2, Q3, Q4, FY
    period_key VARCHAR(20),     -- e.g., "2024-Q1", "2024-FY"
    
    -- Amendment tracking
    is_amended BOOLEAN DEFAULT FALSE,
    amendment_number INTEGER,
    base_form_type VARCHAR(20),  -- 10-K, 10-Q without /A suffix
    supersedes_filing_id INTEGER REFERENCES sec_filings(id),
    
    -- Metadata
    created_at TIMESTAMP DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT uk_sec_filings_accession UNIQUE(accession_number)
);

-- Primary Company Facts store using cache manager pattern
CREATE TABLE IF NOT EXISTS all_companyfacts_store (
    symbol VARCHAR(10) PRIMARY KEY,
    cik INTEGER NOT NULL,
    company_name VARCHAR(255),
    companyfacts JSONB NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    CONSTRAINT unique_cik UNIQUE (cik)
);

-- Complete submission data storage
CREATE TABLE IF NOT EXISTS all_submission_store (
    symbol VARCHAR(10) PRIMARY KEY,
    cik VARCHAR(10) NOT NULL,
    company_name VARCHAR(255),
    submissions_data JSONB NOT NULL,
    latest_filing_date DATE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    CONSTRAINT uk_all_submission_store_cik UNIQUE(cik)
);

-- Quarterly metrics from Frame API
CREATE TABLE IF NOT EXISTS quarterly_metrics (
    id SERIAL PRIMARY KEY,
    cik VARCHAR(10) NOT NULL,
    ticker VARCHAR(10) NOT NULL,
    fiscal_year INTEGER NOT NULL,
    fiscal_period VARCHAR(10) NOT NULL,
    form_type VARCHAR(10),
    category VARCHAR(50) NOT NULL,
    concept_data JSONB NOT NULL,
    common_metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(cik, fiscal_year, fiscal_period, category)
);

-- Quarterly AI summaries
CREATE TABLE IF NOT EXISTS quarterly_ai_summaries (
    id SERIAL PRIMARY KEY,
    cik VARCHAR(10) NOT NULL,
    ticker VARCHAR(10) NOT NULL,
    fiscal_year INTEGER NOT NULL,
    fiscal_period VARCHAR(10) NOT NULL,
    form_type VARCHAR(10),
    financial_summary TEXT,
    ai_analysis JSONB,
    scores JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(cik, fiscal_year, fiscal_period)
);

-- ================================================================================================
-- LLM RESPONSE STORE - PRIMARY TABLE FOR ALL AI INTERACTIONS
-- ================================================================================================

-- Core table for storing all LLM prompts and responses
-- llm_type values: 'sec' (fundamental), 'ta' (technical), 'full' (synthesis)
CREATE TABLE IF NOT EXISTS llm_response_store (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    form_type VARCHAR(10) NOT NULL DEFAULT 'N/A',
    period VARCHAR(10) NOT NULL DEFAULT 'N/A',
    prompt_context JSONB NOT NULL,
    model_info JSONB NOT NULL,
    response JSONB NOT NULL,
    metadata JSONB,
    llm_type VARCHAR(10) NOT NULL DEFAULT 'sec',
    ts TIMESTAMP DEFAULT NOW(),
    CONSTRAINT llm_response_store_symbol_form_type_period_llm_type_key 
        UNIQUE (symbol, form_type, period, llm_type)
);

-- SEC API response caching
CREATE TABLE IF NOT EXISTS sec_response_store (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    category VARCHAR(50) NOT NULL,
    period VARCHAR(10) NOT NULL,
    response JSONB NOT NULL,
    metadata JSONB,
    ts TIMESTAMP DEFAULT NOW(),
    CONSTRAINT sec_response_store_symbol_category_period_key 
        UNIQUE (symbol, category, period)
);

-- SEC category store for tracking concept extraction
CREATE TABLE IF NOT EXISTS sec_category_store (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    category VARCHAR(50) NOT NULL,
    period VARCHAR(10) NOT NULL,
    fiscal_year INTEGER NOT NULL,
    total_concepts INTEGER DEFAULT 0,
    successful_concepts INTEGER DEFAULT 0,
    concept_details JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    CONSTRAINT uk_sec_category_store UNIQUE(symbol, category, period, fiscal_year)
);

-- ================================================================================================
-- ANALYSIS TABLES
-- ================================================================================================

-- Stock analysis results (SQLAlchemy model)
CREATE TABLE IF NOT EXISTS stock_analysis (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    analysis_date DATE NOT NULL,
    fundamental_score NUMERIC,
    technical_score NUMERIC,
    overall_score NUMERIC,
    recommendation VARCHAR(20),
    risk_assessment TEXT,
    report_summary TEXT,
    fundamental_metrics JSONB,
    technical_indicators JSONB,
    combined_analysis JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(symbol, analysis_date)
);

-- Technical indicators (SQLAlchemy model)
CREATE TABLE IF NOT EXISTS technical_indicators (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    analysis_date DATE NOT NULL,
    current_price NUMERIC,
    sma_20 NUMERIC,
    sma_50 NUMERIC,
    sma_200 NUMERIC,
    rsi NUMERIC,
    macd NUMERIC,
    volume BIGINT,
    volatility NUMERIC,
    trend_direction VARCHAR(10),
    support_level NUMERIC,
    resistance_level NUMERIC,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(symbol, analysis_date)
);

-- Analysis reports metadata (SQLAlchemy model)
CREATE TABLE IF NOT EXISTS analysis_reports (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    report_type VARCHAR(50) NOT NULL,
    file_path TEXT,
    generated_at TIMESTAMP DEFAULT NOW(),
    file_size BIGINT,
    status VARCHAR(20) DEFAULT 'generated',
    created_at TIMESTAMP DEFAULT NOW()
);

-- ================================================================================================
-- SCHEMA VERSION TRACKING
-- ================================================================================================

-- Schema version table for migration tracking
CREATE TABLE IF NOT EXISTS schema_version (
    version VARCHAR(20) PRIMARY KEY,
    applied_at TIMESTAMP DEFAULT NOW(),
    description TEXT
);

-- ================================================================================================
-- INDEXES FOR PERFORMANCE
-- ================================================================================================

-- Ticker CIK mapping indexes
CREATE INDEX IF NOT EXISTS idx_ticker_cik_mapping_cik ON ticker_cik_mapping(cik);
CREATE INDEX IF NOT EXISTS idx_ticker_cik_mapping_company ON ticker_cik_mapping(company_name);

-- SEC submissions v2 indexes
CREATE INDEX IF NOT EXISTS idx_sec_submissions_v2_ticker ON sec_submissions_v2(ticker);
CREATE INDEX IF NOT EXISTS idx_sec_submissions_v2_company ON sec_submissions_v2(company_name);
CREATE INDEX IF NOT EXISTS idx_sec_submissions_v2_sic ON sec_submissions_v2(sic);
CREATE INDEX IF NOT EXISTS idx_sec_submissions_v2_updated ON sec_submissions_v2(updated_at);
CREATE INDEX IF NOT EXISTS idx_sec_submissions_v2_parsed_gin ON sec_submissions_v2 USING GIN (parsed_data);

-- SEC filings indexes
CREATE INDEX IF NOT EXISTS idx_sec_filings_cik ON sec_filings(cik);
CREATE INDEX IF NOT EXISTS idx_sec_filings_ticker ON sec_filings(ticker);
CREATE INDEX IF NOT EXISTS idx_sec_filings_form_type ON sec_filings(form_type);
CREATE INDEX IF NOT EXISTS idx_sec_filings_filing_date ON sec_filings(filing_date DESC);
CREATE INDEX IF NOT EXISTS idx_sec_filings_fiscal ON sec_filings(fiscal_year DESC, fiscal_period);
CREATE INDEX IF NOT EXISTS idx_sec_filings_period_key ON sec_filings(period_key);
CREATE INDEX IF NOT EXISTS idx_sec_filings_base_form ON sec_filings(base_form_type);
CREATE INDEX IF NOT EXISTS idx_sec_filings_amended ON sec_filings(is_amended) WHERE is_amended = TRUE;

-- Partial unique index for non-amended filings per period
CREATE UNIQUE INDEX IF NOT EXISTS uk_sec_filings_period_partial 
ON sec_filings(cik, fiscal_year, fiscal_period, form_type)
WHERE is_amended = FALSE;

-- Company Facts store indexes
CREATE INDEX IF NOT EXISTS idx_all_companyfacts_store_symbol ON all_companyfacts_store(symbol);
CREATE INDEX IF NOT EXISTS idx_all_companyfacts_store_cik ON all_companyfacts_store(cik);
CREATE INDEX IF NOT EXISTS idx_all_companyfacts_store_updated ON all_companyfacts_store(updated_at);
CREATE INDEX IF NOT EXISTS idx_all_companyfacts_store_company_name ON all_companyfacts_store(company_name);

-- All submission store indexes
CREATE INDEX IF NOT EXISTS idx_all_submission_store_symbol ON all_submission_store(symbol);
CREATE INDEX IF NOT EXISTS idx_all_submission_store_cik ON all_submission_store(cik);
CREATE INDEX IF NOT EXISTS idx_all_submission_store_updated ON all_submission_store(updated_at);
CREATE INDEX IF NOT EXISTS idx_all_submission_store_latest_filing ON all_submission_store(latest_filing_date DESC);

-- Quarterly metrics indexes
CREATE INDEX IF NOT EXISTS idx_quarterly_metrics_ticker ON quarterly_metrics(ticker);
CREATE INDEX IF NOT EXISTS idx_quarterly_metrics_cik_year_period ON quarterly_metrics(cik, fiscal_year, fiscal_period);
CREATE INDEX IF NOT EXISTS idx_quarterly_metrics_category ON quarterly_metrics(category);
CREATE INDEX IF NOT EXISTS idx_quarterly_metrics_created ON quarterly_metrics(created_at);

-- Quarterly AI summaries indexes
CREATE INDEX IF NOT EXISTS idx_quarterly_ai_summaries_ticker ON quarterly_ai_summaries(ticker);
CREATE INDEX IF NOT EXISTS idx_quarterly_ai_summaries_cik_year_period ON quarterly_ai_summaries(cik, fiscal_year, fiscal_period);
CREATE INDEX IF NOT EXISTS idx_quarterly_ai_summaries_created ON quarterly_ai_summaries(created_at);

-- LLM response store indexes (CRITICAL FOR PERFORMANCE)
CREATE INDEX IF NOT EXISTS idx_llm_response_store_symbol ON llm_response_store(symbol);
CREATE INDEX IF NOT EXISTS idx_llm_response_store_llm_type ON llm_response_store(llm_type);
CREATE INDEX IF NOT EXISTS idx_llm_response_store_symbol_llm_type ON llm_response_store(symbol, llm_type);
CREATE INDEX IF NOT EXISTS idx_llm_response_store_ts ON llm_response_store(ts);
CREATE INDEX IF NOT EXISTS idx_llm_response_store_symbol_form_period ON llm_response_store(symbol, form_type, period);

-- SEC response store indexes
CREATE INDEX IF NOT EXISTS idx_sec_response_store_symbol ON sec_response_store(symbol);
CREATE INDEX IF NOT EXISTS idx_sec_response_store_category ON sec_response_store(category);
CREATE INDEX IF NOT EXISTS idx_sec_response_store_symbol_category ON sec_response_store(symbol, category);
CREATE INDEX IF NOT EXISTS idx_sec_response_store_ts ON sec_response_store(ts);

-- SEC category store indexes
CREATE INDEX IF NOT EXISTS idx_sec_category_store_symbol ON sec_category_store(symbol);
CREATE INDEX IF NOT EXISTS idx_sec_category_store_category ON sec_category_store(category);
CREATE INDEX IF NOT EXISTS idx_sec_category_store_updated ON sec_category_store(updated_at);

-- Stock analysis indexes
CREATE INDEX IF NOT EXISTS idx_stock_analysis_symbol ON stock_analysis(symbol);
CREATE INDEX IF NOT EXISTS idx_stock_analysis_date ON stock_analysis(analysis_date);
CREATE INDEX IF NOT EXISTS idx_stock_analysis_score ON stock_analysis(overall_score);
CREATE INDEX IF NOT EXISTS idx_stock_analysis_recommendation ON stock_analysis(recommendation);
CREATE INDEX IF NOT EXISTS idx_stock_analysis_created ON stock_analysis(created_at);

-- Technical indicators indexes
CREATE INDEX IF NOT EXISTS idx_technical_indicators_symbol ON technical_indicators(symbol);
CREATE INDEX IF NOT EXISTS idx_technical_indicators_date ON technical_indicators(analysis_date);
CREATE INDEX IF NOT EXISTS idx_technical_indicators_created ON technical_indicators(created_at);

-- Analysis reports indexes
CREATE INDEX IF NOT EXISTS idx_analysis_reports_symbol ON analysis_reports(symbol);
CREATE INDEX IF NOT EXISTS idx_analysis_reports_type ON analysis_reports(report_type);
CREATE INDEX IF NOT EXISTS idx_analysis_reports_status ON analysis_reports(status);
CREATE INDEX IF NOT EXISTS idx_analysis_reports_generated ON analysis_reports(generated_at);

-- ================================================================================================
-- VIEWS FOR ANALYSIS
-- ================================================================================================

-- Latest recommendations view
CREATE OR REPLACE VIEW latest_recommendations AS
SELECT DISTINCT ON (stock_analysis.symbol) 
    stock_analysis.symbol,
    stock_analysis.analysis_date,
    stock_analysis.fundamental_score,
    stock_analysis.technical_score,
    stock_analysis.overall_score,
    stock_analysis.recommendation,
    stock_analysis.risk_assessment,
    stock_analysis.report_summary
FROM stock_analysis
ORDER BY stock_analysis.symbol, stock_analysis.analysis_date DESC;

-- Table size information view
CREATE OR REPLACE VIEW table_size_info AS
SELECT 
    t.schemaname,
    t.tablename,
    pg_size_pretty(pg_total_relation_size(((((t.schemaname)::text || '.'::text) || (t.tablename)::text))::regclass)) AS size,
    pg_total_relation_size(((((t.schemaname)::text || '.'::text) || (t.tablename)::text))::regclass) AS size_bytes,
    COALESCE(s.n_tup_ins, (0)::bigint) AS total_rows,
    COALESCE(s.n_tup_upd, (0)::bigint) AS updates,
    COALESCE(s.n_tup_del, (0)::bigint) AS deletes,
    COALESCE(s.last_analyze, s.last_autoanalyze) AS last_analyzed
FROM (pg_tables t
    LEFT JOIN pg_stat_user_tables s ON (((s.relname = t.tablename) AND (s.schemaname = t.schemaname))))
WHERE (t.schemaname = 'public'::name)
ORDER BY (pg_total_relation_size(((((t.schemaname)::text || '.'::text) || (t.tablename)::text))::regclass)) DESC;

-- SEC category freshness view
CREATE OR REPLACE VIEW v_sec_category_freshness AS
SELECT 
    sec_category_store.symbol,
    sec_category_store.category,
    max(sec_category_store.updated_at) AS last_updated,
    count(DISTINCT (((sec_category_store.period)::text || '_'::text) || sec_category_store.fiscal_year)) AS periods_available,
    avg(((sec_category_store.successful_concepts)::double precision / (NULLIF(sec_category_store.total_concepts, 0))::double precision)) AS avg_success_rate
FROM sec_category_store
GROUP BY sec_category_store.symbol, sec_category_store.category
ORDER BY sec_category_store.symbol, sec_category_store.category;

-- LLM response summary view for monitoring
CREATE OR REPLACE VIEW llm_response_summary AS
SELECT 
    symbol,
    llm_type,
    COUNT(*) as response_count,
    MAX(ts) as latest_response,
    MIN(ts) as first_response,
    AVG((metadata->>'processing_time_ms')::numeric) as avg_processing_time_ms,
    SUM((metadata->>'response_length')::numeric) as total_response_length
FROM llm_response_store 
WHERE metadata IS NOT NULL
GROUP BY symbol, llm_type
ORDER BY symbol, llm_type;

-- Quarterly performance view
CREATE OR REPLACE VIEW quarterly_performance AS
SELECT 
    qa.ticker,
    qa.fiscal_year,
    qa.fiscal_period,
    qa.form_type,
    (qa.scores->>'financial_health')::numeric as financial_health,
    (qa.scores->>'growth_prospects')::numeric as growth_prospects,
    (qa.scores->>'business_quality')::numeric as business_quality,
    qa.created_at
FROM quarterly_ai_summaries qa
ORDER BY qa.ticker, qa.fiscal_year DESC, qa.fiscal_period;

-- ================================================================================================
-- MATERIALIZED VIEWS
-- ================================================================================================

-- Earnings submission materialized view for fast earnings lookup
CREATE MATERIALIZED VIEW IF NOT EXISTS earnings_submission_materialized_view AS
SELECT DISTINCT ON (sf.cik, sf.fiscal_year, sf.fiscal_period, sf.base_form_type)
    sf.id as filing_id,
    sf.cik,
    sf.ticker,
    sf.form_type,
    sf.filing_date,
    sf.accession_number,
    sf.fiscal_year,
    sf.fiscal_period,
    sf.period_key,
    sf.is_amended,
    sf.amendment_number,
    sf.base_form_type,
    sv2.company_name
FROM sec_filings sf
LEFT JOIN sec_submissions_v2 sv2 ON sf.cik = sv2.cik
WHERE sf.base_form_type IN ('10-K', '10-Q')
ORDER BY sf.cik, sf.fiscal_year, sf.fiscal_period, sf.base_form_type,
         sf.is_amended DESC, sf.amendment_number DESC NULLS LAST, sf.filing_date DESC;

-- Create unique index on materialized view
CREATE UNIQUE INDEX IF NOT EXISTS idx_earnings_mv_unique 
ON earnings_submission_materialized_view(cik, fiscal_year, fiscal_period, base_form_type);

-- ================================================================================================
-- FUNCTIONS
-- ================================================================================================

-- Function to get latest filing for a period (handles amendments)
CREATE OR REPLACE FUNCTION get_latest_filing_for_period(
    p_ticker VARCHAR,
    p_fiscal_year INTEGER,
    p_fiscal_period VARCHAR
)
RETURNS TABLE (
    filing_id INTEGER,
    form_type VARCHAR,
    filing_date DATE,
    accession_number VARCHAR,
    is_amended BOOLEAN,
    amendment_number INTEGER
) AS $$
BEGIN
    RETURN QUERY
    WITH ranked_filings AS (
        SELECT 
            f.id as filing_id,
            f.form_type,
            f.filing_date,
            f.accession_number,
            f.is_amended,
            f.amendment_number,
            ROW_NUMBER() OVER (
                PARTITION BY f.base_form_type 
                ORDER BY 
                    f.is_amended DESC,  -- Amendments first
                    f.amendment_number DESC NULLS LAST,  -- Higher amendments first
                    f.filing_date DESC  -- Most recent first
            ) as rn
        FROM sec_filings f
        WHERE f.ticker = p_ticker
          AND f.fiscal_year = p_fiscal_year
          AND f.fiscal_period = p_fiscal_period
    )
    SELECT 
        filing_id,
        form_type,
        filing_date,
        accession_number,
        is_amended,
        amendment_number
    FROM ranked_filings
    WHERE rn = 1;
END;
$$ LANGUAGE plpgsql;

-- Function to get recent earnings filings with amendment handling
CREATE OR REPLACE FUNCTION get_recent_earnings_filings(
    p_ticker VARCHAR,
    p_limit INTEGER DEFAULT 8
)
RETURNS TABLE (
    filing_id INTEGER,
    form_type VARCHAR,
    filing_date DATE,
    accession_number VARCHAR,
    fiscal_year INTEGER,
    fiscal_period VARCHAR,
    period_key VARCHAR,
    is_amended BOOLEAN
) AS $$
BEGIN
    RETURN QUERY
    WITH latest_per_period AS (
        SELECT DISTINCT ON (f.period_key)
            f.id as filing_id,
            f.form_type,
            f.filing_date,
            f.accession_number,
            f.fiscal_year,
            f.fiscal_period,
            f.period_key,
            f.is_amended
        FROM sec_filings f
        WHERE f.ticker = p_ticker
          AND f.base_form_type IN ('10-K', '10-Q')
        ORDER BY 
            f.period_key,
            f.is_amended DESC,  -- Amendments first
            f.amendment_number DESC NULLS LAST,
            f.filing_date DESC
    )
    SELECT * FROM latest_per_period
    ORDER BY fiscal_year DESC, fiscal_period DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- Function to clean old cache data
CREATE OR REPLACE FUNCTION cleanup_old_cache_data(days_to_keep INTEGER DEFAULT 30)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    -- Clean old Company Facts data (keep fresh for better performance)
    DELETE FROM all_companyfacts_store 
    WHERE updated_at < NOW() - INTERVAL '1 day' * (days_to_keep / 2);
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    
    -- Note: Do NOT delete from llm_response_store as these are permanent for observability
    
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Function to get LLM performance statistics
CREATE OR REPLACE FUNCTION get_llm_performance_stats()
RETURNS TABLE(
    llm_type VARCHAR,
    total_responses BIGINT,
    avg_processing_time_ms NUMERIC,
    avg_response_length NUMERIC,
    latest_response TIMESTAMP
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        lrs.llm_type::VARCHAR,
        COUNT(*)::BIGINT as total_responses,
        AVG((lrs.metadata->>'processing_time_ms')::numeric) as avg_processing_time_ms,
        AVG((lrs.metadata->>'response_length')::numeric) as avg_response_length,
        MAX(lrs.ts) as latest_response
    FROM llm_response_store lrs
    WHERE lrs.metadata IS NOT NULL
    GROUP BY lrs.llm_type
    ORDER BY lrs.llm_type;
END;
$$ LANGUAGE plpgsql;

-- ================================================================================================
-- JSONB COMPRESSION CONFIGURATION
-- ================================================================================================

-- Enable JSONB compression on PostgreSQL 14+
ALTER TABLE sec_submissions_v2 ALTER COLUMN parsed_data SET STORAGE EXTENDED;
ALTER TABLE sec_submissions_v2 ALTER COLUMN raw_data SET STORAGE EXTENDED;

-- All Company Facts Store - Enable compression
ALTER TABLE all_companyfacts_store 
    ALTER COLUMN companyfacts SET STORAGE EXTENDED,
    ALTER COLUMN metadata SET STORAGE EXTENDED;

-- All Submission Store (large consolidated data)
ALTER TABLE all_submission_store 
    ALTER COLUMN submissions_data SET STORAGE EXTENDED;

-- Quarterly Metrics - Enable compression
ALTER TABLE quarterly_metrics 
    ALTER COLUMN concept_data SET STORAGE EXTENDED,
    ALTER COLUMN common_metadata SET STORAGE EXTENDED;

-- Quarterly AI Summaries
ALTER TABLE quarterly_ai_summaries 
    ALTER COLUMN ai_analysis SET STORAGE EXTENDED,
    ALTER COLUMN scores SET STORAGE EXTENDED;

-- LLM Response Store - Critical for large AI responses
ALTER TABLE llm_response_store 
    ALTER COLUMN prompt_context SET STORAGE EXTENDED,
    ALTER COLUMN model_info SET STORAGE EXTENDED,
    ALTER COLUMN response SET STORAGE EXTENDED,
    ALTER COLUMN metadata SET STORAGE EXTENDED;

-- SEC Response Store
ALTER TABLE sec_response_store 
    ALTER COLUMN response SET STORAGE EXTENDED,
    ALTER COLUMN metadata SET STORAGE EXTENDED;

-- Stock Analysis Results
ALTER TABLE stock_analysis 
    ALTER COLUMN fundamental_metrics SET STORAGE EXTENDED,
    ALTER COLUMN technical_indicators SET STORAGE EXTENDED,
    ALTER COLUMN combined_analysis SET STORAGE EXTENDED;

-- ================================================================================================
-- CONSTRAINTS AND FOREIGN KEYS
-- ================================================================================================

-- Add foreign key constraints where appropriate
ALTER TABLE sec_submissions_v2 
ADD CONSTRAINT fk_sec_submissions_v2_ticker 
FOREIGN KEY (ticker) REFERENCES ticker_cik_mapping(ticker) 
ON DELETE CASCADE;

ALTER TABLE all_companyfacts_store 
ADD CONSTRAINT fk_all_companyfacts_store_symbol 
FOREIGN KEY (symbol) REFERENCES ticker_cik_mapping(ticker) 
ON DELETE CASCADE;

ALTER TABLE all_submission_store 
ADD CONSTRAINT fk_all_submission_store_symbol 
FOREIGN KEY (symbol) REFERENCES ticker_cik_mapping(ticker) 
ON DELETE CASCADE;

ALTER TABLE quarterly_metrics 
ADD CONSTRAINT fk_quarterly_metrics_ticker 
FOREIGN KEY (ticker) REFERENCES ticker_cik_mapping(ticker) 
ON DELETE CASCADE;

-- ================================================================================================
-- TABLE COMMENTS FOR DOCUMENTATION
-- ================================================================================================

COMMENT ON TABLE ticker_cik_mapping IS 'Maps stock tickers to SEC CIK numbers using SEC ticker.txt';
COMMENT ON TABLE sec_submissions_v2 IS 'Enhanced SEC submissions storage with parsed data from submission_processor';
COMMENT ON TABLE sec_filings IS 'Individual SEC filings with amendment tracking and fiscal period mapping';
COMMENT ON TABLE all_companyfacts_store IS 'Primary Company Facts store using cache manager pattern - stores complete SEC companyfacts.json for each symbol';
COMMENT ON TABLE all_submission_store IS 'Complete submission data storage for large consolidated filings';
COMMENT ON TABLE quarterly_metrics IS 'Financial metrics extracted from SEC Frame API by quarter';
COMMENT ON TABLE quarterly_ai_summaries IS 'AI-generated summaries for each quarter';
COMMENT ON TABLE llm_response_store IS 'Primary table storing ALL LLM prompts and responses for observability';
COMMENT ON TABLE sec_response_store IS 'SEC API response caching for category-based data extraction';
COMMENT ON TABLE sec_category_store IS 'Tracks SEC concept extraction by category and period';
COMMENT ON TABLE stock_analysis IS 'Final stock analysis results with scores and recommendations';
COMMENT ON TABLE technical_indicators IS 'Technical analysis results from market data';
COMMENT ON TABLE analysis_reports IS 'Metadata for generated analysis reports';

-- Column comments for key tables
COMMENT ON COLUMN llm_response_store.llm_type IS 'Type of LLM analysis: sec (fundamental), ta (technical), full (synthesis)';
COMMENT ON COLUMN llm_response_store.form_type IS 'SEC form type (10-K, 10-Q, etc.) - N/A for technical analysis';
COMMENT ON COLUMN llm_response_store.period IS 'Fiscal period (Q1, Q2, Q3, FY) - N/A for technical analysis';
COMMENT ON COLUMN llm_response_store.prompt_context IS 'Complete prompt sent to LLM including context';
COMMENT ON COLUMN llm_response_store.model_info IS 'LLM model name, version, and configuration';
COMMENT ON COLUMN llm_response_store.response IS 'Complete LLM response in structured format';
COMMENT ON COLUMN llm_response_store.metadata IS 'Processing time, response length, timestamps, etc.';

COMMENT ON COLUMN sec_submissions_v2.parsed_data IS 'Parsed submission data from submission_processor.parse_submissions()';
COMMENT ON COLUMN sec_submissions_v2.raw_data IS 'Original SEC API response (compressed JSONB)';

COMMENT ON COLUMN sec_filings.period_key IS 'Standardized period identifier (YYYY-FP format, e.g., 2024-Q1)';
COMMENT ON COLUMN sec_filings.supersedes_filing_id IS 'References the filing that this amendment supersedes';

-- ================================================================================================
-- DATABASE CONFIGURATION FOR OPTIMAL PERFORMANCE
-- ================================================================================================

-- Set up database configuration for optimal performance
ALTER SYSTEM SET shared_preload_libraries = 'pg_stat_statements';
ALTER SYSTEM SET max_connections = 100;
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET default_statistics_target = 100;

-- ================================================================================================
-- INITIAL DATA AND SCHEMA VERSION
-- ================================================================================================

-- Insert current schema version
INSERT INTO schema_version (version, description) 
VALUES ('3.0.0', 'Consolidated InvestiGator schema - cleaned and optimized')
ON CONFLICT (version) DO NOTHING;

-- ================================================================================================
-- ANALYZE TABLES FOR OPTIMIZER
-- ================================================================================================

-- After schema creation, analyze tables to update statistics
ANALYZE ticker_cik_mapping;
ANALYZE sec_submissions_v2;
ANALYZE sec_filings;
ANALYZE all_companyfacts_store;
ANALYZE all_submission_store;
ANALYZE quarterly_metrics;
ANALYZE quarterly_ai_summaries;
ANALYZE llm_response_store;
ANALYZE sec_response_store;
ANALYZE sec_category_store;
ANALYZE stock_analysis;
ANALYZE technical_indicators;
ANALYZE analysis_reports;

-- Refresh materialized view
REFRESH MATERIALIZED VIEW earnings_submission_materialized_view;

-- ================================================================================================
-- COMPLETION MESSAGE
-- ================================================================================================

DO $$
BEGIN
    RAISE NOTICE '🐊 InvestiGator consolidated database schema installation completed successfully! 🚀';
    RAISE NOTICE 'Schema version: 3.0.0 (Consolidated)';
    RAISE NOTICE 'Tables created: %, Indexes: %, Views: %, Functions: %', 
        (SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public'),
        (SELECT COUNT(*) FROM pg_indexes WHERE schemaname = 'public'),
        (SELECT COUNT(*) FROM information_schema.views WHERE table_schema = 'public'),
        (SELECT COUNT(*) FROM information_schema.routines WHERE routine_schema = 'public');
    RAISE NOTICE '✅ All obsolete tables and objects have been removed';
    RAISE NOTICE '🗜️ JSONB compression enabled for optimal storage';
    RAISE NOTICE '📊 Ready for production use!';
END $$;