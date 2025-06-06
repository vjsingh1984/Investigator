-- PostgreSQL Native Compression Configuration
-- This ensures all JSONB columns use PostgreSQL's native TOAST compression
-- TOAST (The Oversized-Attribute Storage Technique) automatically compresses large values

-- ================================================================================================
-- CONFIGURE JSONB COMPRESSION FOR ALL TABLES
-- ================================================================================================

-- SEC Submissions - Enable compression on JSONB columns
ALTER TABLE sec_submissions 
    ALTER COLUMN filings SET STORAGE EXTENDED,
    ALTER COLUMN recent_filings SET STORAGE EXTENDED;

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

-- Technical Analysis Results
ALTER TABLE technical_analysis 
    ALTER COLUMN technical_data SET STORAGE EXTENDED,
    ALTER COLUMN market_data SET STORAGE EXTENDED;

-- Stock Analysis Results
ALTER TABLE stock_analysis 
    ALTER COLUMN fundamental_metrics SET STORAGE EXTENDED,
    ALTER COLUMN technical_indicators SET STORAGE EXTENDED,
    ALTER COLUMN combined_analysis SET STORAGE EXTENDED;

-- Technical Indicators Store
ALTER TABLE technical_indicators_store 
    ALTER COLUMN indicators_data SET STORAGE EXTENDED,
    ALTER COLUMN metadata SET STORAGE EXTENDED;

-- Synthesis Results
ALTER TABLE synthesis_results 
    ALTER COLUMN fundamental_data SET STORAGE EXTENDED,
    ALTER COLUMN technical_data SET STORAGE EXTENDED,
    ALTER COLUMN synthesis SET STORAGE EXTENDED,
    ALTER COLUMN recommendation SET STORAGE EXTENDED;

-- ================================================================================================
-- STORAGE OPTIONS EXPLAINED:
-- ================================================================================================
-- PLAIN: No compression, no out-of-line storage
-- EXTERNAL: No compression, but out-of-line storage allowed
-- EXTENDED: Compression and out-of-line storage (DEFAULT for JSONB - best for our use case)
-- MAIN: Compression, but prefer inline storage

-- ================================================================================================
-- VERIFY COMPRESSION SETTINGS
-- ================================================================================================
-- Run this query to verify all JSONB columns are using EXTENDED storage:

/*
SELECT 
    n.nspname AS schema_name,
    c.relname AS table_name,
    a.attname AS column_name,
    t.typname AS data_type,
    CASE a.attstorage
        WHEN 'p' THEN 'PLAIN'
        WHEN 'e' THEN 'EXTERNAL'
        WHEN 'x' THEN 'EXTENDED'
        WHEN 'm' THEN 'MAIN'
    END AS storage_type
FROM pg_attribute a
JOIN pg_class c ON a.attrelid = c.oid
JOIN pg_namespace n ON c.relnamespace = n.oid
JOIN pg_type t ON a.atttypid = t.oid
WHERE n.nspname = 'public'
    AND a.attnum > 0
    AND NOT a.attisdropped
    AND t.typname = 'jsonb'
ORDER BY c.relname, a.attnum;
*/

-- ================================================================================================
-- ANALYZE TABLES FOR OPTIMIZER
-- ================================================================================================
-- After changing storage settings, analyze tables to update statistics
ANALYZE sec_submissions;
ANALYZE all_companyfacts_store;
ANALYZE all_submission_store;
ANALYZE quarterly_metrics;
ANALYZE quarterly_ai_summaries;
ANALYZE llm_response_store;
ANALYZE sec_response_store;
ANALYZE technical_analysis;
ANALYZE stock_analysis;
ANALYZE technical_indicators_store;
ANALYZE synthesis_results;