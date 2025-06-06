-- Data Validation Queries for SEC Submissions (Hash Partitioned Schema)
-- Run these queries after processing CIKs to validate the data

-- 1. Basic statistics across all partitions
SELECT 
    COUNT(*) as total_records,
    COUNT(DISTINCT cik) as unique_ciks,
    MIN(filing_date) as earliest_filing,
    MAX(filing_date) as latest_filing,
    MAX(created_at) as last_processed,
    pg_size_pretty(pg_total_relation_size('submissions')) as total_size
FROM submissions;

-- 2. Check hash partition distribution
SELECT 
    partition_name,
    cik_count,
    filing_count,
    size_mb,
    avg_filings_per_cik,
    ROUND(100.0 * cik_count / SUM(cik_count) OVER(), 2) as cik_percentage,
    ROUND(100.0 * filing_count / SUM(filing_count) OVER(), 2) as filing_percentage
FROM get_partition_stats()
ORDER BY filing_count DESC;

-- 3. Validate entity synchronization
SELECT 
    'Entities table' as table_name,
    COUNT(*) as count
FROM entities

UNION ALL

SELECT 
    'Unique CIKs in submissions' as table_name,
    COUNT(DISTINCT cik) as count
FROM submissions

UNION ALL

SELECT 
    'CIKs missing from entities' as table_name,
    COUNT(DISTINCT s.cik) as count
FROM submissions s
LEFT JOIN entities e ON s.cik = e.cik
WHERE e.cik IS NULL AND s.entity_name IS NOT NULL;

-- 4. Check materialized view freshness
SELECT 
    'recent_activity' as view_name,
    COUNT(*) as record_count,
    MIN(filing_date) as earliest_date,
    MAX(filing_date) as latest_date
FROM recent_activity

UNION ALL

SELECT 
    'entity_stats' as view_name,
    COUNT(*) as record_count,
    MIN(earliest_filing) as earliest_date,
    MAX(latest_filing) as latest_date
FROM entity_stats;

-- 5. Form type distribution with percentages
SELECT 
    form_type,
    COUNT(*) as count,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 2) as percentage,
    COUNT(DISTINCT cik) as unique_entities
FROM submissions 
WHERE form_type IS NOT NULL
GROUP BY form_type
ORDER BY count DESC
LIMIT 20;

-- 6. Recent processing activity (last 24 hours)
SELECT 
    DATE_TRUNC('hour', created_at) as hour,
    COUNT(*) as filings_processed,
    COUNT(DISTINCT cik) as entities_processed
FROM submissions 
WHERE created_at >= CURRENT_TIMESTAMP - INTERVAL '24 hours'
GROUP BY DATE_TRUNC('hour', created_at)
ORDER BY hour DESC;

-- 7. Data quality checks
SELECT 
    'Missing form_type' as issue,
    COUNT(*) as count,
    ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM submissions), 2) as percentage
FROM submissions 
WHERE form_type IS NULL OR form_type = ''

UNION ALL

SELECT 
    'Missing filing_date' as issue,
    COUNT(*) as count,
    ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM submissions), 2) as percentage
FROM submissions 
WHERE filing_date IS NULL

UNION ALL

SELECT 
    'Future filing_date' as issue,
    COUNT(*) as count,
    ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM submissions), 2) as percentage
FROM submissions 
WHERE filing_date > CURRENT_DATE

UNION ALL

SELECT 
    'Very old filings (before 1990)' as issue,
    COUNT(*) as count,
    ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM submissions), 2) as percentage
FROM submissions 
WHERE filing_date < '1990-01-01'

UNION ALL

SELECT 
    'Missing entity_name with recent filings' as issue,
    COUNT(*) as count,
    ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM submissions), 2) as percentage
FROM submissions 
WHERE entity_name IS NULL AND filing_date >= CURRENT_DATE - INTERVAL '1 year';

-- 8. Sample data across different partitions
WITH partition_samples AS (
    SELECT 
        cik,
        entity_name,
        form_type,
        filing_date,
        accession_number,
        LPAD(cik::text, 10, '0') as formatted_cik,
        size_bytes,
        is_xbrl,
        ABS(HASHTEXT(cik::text)) % 16 as calculated_partition
    FROM submissions 
    WHERE entity_name IS NOT NULL
    ORDER BY created_at DESC 
    LIMIT 20
)
SELECT 
    calculated_partition,
    cik,
    formatted_cik,
    entity_name,
    form_type,
    filing_date,
    pg_size_pretty(size_bytes) as size
FROM partition_samples
ORDER BY calculated_partition, filing_date DESC;

-- 9. Entity metadata completeness analysis
SELECT 
    COUNT(*) as total_entities,
    COUNT(entity_name) as has_entity_name,
    COUNT(sic) as has_sic,
    COUNT(tickers) as has_tickers,
    COUNT(website) as has_website,
    COUNT(addresses) as has_addresses,
    ROUND(100.0 * COUNT(entity_name) / COUNT(*), 1) as name_completeness,
    ROUND(100.0 * COUNT(sic) / COUNT(*), 1) as sic_completeness,
    ROUND(100.0 * COUNT(tickers) / COUNT(*), 1) as ticker_completeness,
    ROUND(100.0 * COUNT(website) / COUNT(*), 1) as website_completeness
FROM entities;

-- 10. Top entities by filing volume
SELECT 
    e.cik,
    LPAD(e.cik::text, 10, '0') as formatted_cik,
    e.entity_name,
    es.total_filings,
    es.major_forms,
    es.recent_filings,
    es.latest_filing,
    ARRAY_TO_STRING(e.tickers, ', ') as ticker_symbols,
    pg_size_pretty(es.total_bytes) as total_size
FROM entities e
JOIN entity_stats es ON e.cik = es.cik
ORDER BY es.total_filings DESC
LIMIT 15;

-- 11. Filing trends by year
SELECT 
    EXTRACT(YEAR FROM filing_date) as year,
    COUNT(*) as total_filings,
    COUNT(DISTINCT cik) as unique_entities,
    COUNT(CASE WHEN form_type IN ('10-K', '10-Q', '8-K') THEN 1 END) as major_forms,
    COUNT(CASE WHEN is_xbrl THEN 1 END) as xbrl_filings,
    ROUND(100.0 * COUNT(CASE WHEN is_xbrl THEN 1 END) / COUNT(*), 1) as xbrl_percentage,
    pg_size_pretty(SUM(size_bytes)) as total_size
FROM submissions 
WHERE filing_date >= '2000-01-01'
GROUP BY EXTRACT(YEAR FROM filing_date)
ORDER BY year DESC
LIMIT 10;

-- 12. Index usage statistics
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_tup_read,
    idx_tup_fetch,
    CASE 
        WHEN idx_tup_read > 0 THEN ROUND(100.0 * idx_tup_fetch / idx_tup_read, 2)
        ELSE 0 
    END as hit_rate_percentage
FROM pg_stat_user_indexes 
WHERE tablename IN ('submissions', 'entities') 
   OR tablename LIKE 'submissions_p%'
ORDER BY idx_tup_read DESC
LIMIT 20;

-- 13. Performance validation queries
-- Test partition pruning (should only hit one partition)
EXPLAIN (ANALYZE, BUFFERS) 
SELECT COUNT(*) FROM submissions WHERE cik = 320193;

-- Test recent data query performance  
EXPLAIN (ANALYZE, BUFFERS)
SELECT * FROM recent_activity WHERE entity_name ILIKE '%Apple%' LIMIT 10;

-- Test entity search performance
EXPLAIN (ANALYZE, BUFFERS)
SELECT * FROM search_entities('technology') LIMIT 5;

-- 14. Storage efficiency analysis
SELECT 
    'Database total' as component,
    pg_size_pretty(pg_database_size(current_database())) as size
    
UNION ALL

SELECT 
    'Submissions table' as component,
    pg_size_pretty(pg_total_relation_size('submissions')) as size
    
UNION ALL

SELECT 
    'Entities table' as component,
    pg_size_pretty(pg_total_relation_size('entities')) as size
    
UNION ALL

SELECT 
    'All indexes' as component,
    pg_size_pretty(
        pg_total_relation_size('submissions') - pg_relation_size('submissions') +
        pg_total_relation_size('entities') - pg_relation_size('entities')
    ) as size;

-- 15. Data freshness check
SELECT 
    'Most recent filing' as metric,
    MAX(filing_date)::text as value
FROM submissions

UNION ALL

SELECT 
    'Most recent data load' as metric,
    MAX(created_at)::text as value
FROM submissions

UNION ALL

SELECT 
    'Materialized view: recent_activity age' as metric,
    EXTRACT(EPOCH FROM (NOW() - MAX(created_at)))::text || ' seconds' as value
FROM submissions s
JOIN recent_activity ra ON s.cik = ra.cik AND s.accession_number = ra.accession_number

UNION ALL

SELECT 
    'Entity stats coverage' as metric,
    ROUND(100.0 * COUNT(es.cik) / COUNT(DISTINCT s.cik), 1)::text || '%' as value
FROM submissions s
LEFT JOIN entity_stats es ON s.cik = es.cik
WHERE s.entity_name IS NOT NULL;
