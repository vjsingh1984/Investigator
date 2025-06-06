#!/usr/bin/env python3
"""
Data Access Object for Peer Metrics
Handles database operations for peer group metrics and comparisons
"""

import json
import logging
from datetime import datetime, date
from typing import Dict, List, Optional, Any
from sqlalchemy import text
from utils.db import get_db_manager

logger = logging.getLogger(__name__)


class PeerMetricsDAO:
    """Data Access Object for peer metrics table"""
    
    def __init__(self):
        self.logger = logger
        self.db_manager = get_db_manager()
        
    def save_peer_metrics(self, peer_group_id: str, symbol: str, metric_type: str,
                         metrics_data: Dict[str, Any], sector: str = None, 
                         industry: str = None, peer_symbols: List[str] = None) -> bool:
        """Save peer metrics to database"""
        conn = None
        try:
            conn = self.db_manager.engine.connect()
            
            # Convert metrics_data to JSON string
            metrics_json = json.dumps(metrics_data)
            
            # Convert peer_symbols list to PostgreSQL array format
            peer_symbols_array = '{' + ','.join(peer_symbols) + '}' if peer_symbols else None
            
            # Use current date for calculation_date
            calc_date = date.today()
            
            query = text("""
                INSERT INTO peer_metrics (
                    peer_group_id, symbol, metric_type, sector, industry,
                    metrics_data, peer_symbols, calculation_date
                ) VALUES (
                    :peer_group_id, :symbol, :metric_type, :sector, :industry,
                    :metrics_data::jsonb, :peer_symbols, :calculation_date
                )
                ON CONFLICT (peer_group_id, symbol, metric_type) 
                DO UPDATE SET
                    sector = EXCLUDED.sector,
                    industry = EXCLUDED.industry,
                    metrics_data = EXCLUDED.metrics_data,
                    peer_symbols = EXCLUDED.peer_symbols,
                    calculation_date = EXCLUDED.calculation_date,
                    updated_at = NOW()
            """)
            
            with conn.begin():
                conn.execute(query, {
                    'peer_group_id': peer_group_id,
                    'symbol': symbol,
                    'metric_type': metric_type,
                    'sector': sector,
                    'industry': industry,
                    'metrics_data': metrics_json,
                    'peer_symbols': peer_symbols_array,
                    'calculation_date': calc_date
                })
            
            self.logger.info(f"Saved peer metrics for {symbol} in group {peer_group_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving peer metrics: {e}")
            return False
        finally:
            if conn:
                conn.close()
    
    def get_peer_metrics(self, peer_group_id: str, symbol: str, 
                        metric_type: str) -> Optional[Dict[str, Any]]:
        """Get peer metrics from database"""
        conn = None
        try:
            conn = self.db_manager.engine.connect()
            
            query = text("""
                SELECT 
                    peer_group_id, symbol, metric_type, sector, industry,
                    metrics_data, peer_symbols, calculation_date,
                    created_at, updated_at
                FROM peer_metrics
                WHERE peer_group_id = :peer_group_id 
                  AND symbol = :symbol 
                  AND metric_type = :metric_type
            """)
            
            result = conn.execute(query, {
                'peer_group_id': peer_group_id,
                'symbol': symbol,
                'metric_type': metric_type
            }).fetchone()
            
            if result:
                return {
                    'peer_group_id': result.peer_group_id,
                    'symbol': result.symbol,
                    'metric_type': result.metric_type,
                    'sector': result.sector,
                    'industry': result.industry,
                    'metrics_data': result.metrics_data,
                    'peer_symbols': result.peer_symbols,
                    'calculation_date': result.calculation_date.isoformat() if result.calculation_date else None,
                    'created_at': result.created_at.isoformat() if result.created_at else None,
                    'updated_at': result.updated_at.isoformat() if result.updated_at else None
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting peer metrics: {e}")
            return None
        finally:
            if conn:
                conn.close()
    
    def get_peer_group_metrics(self, peer_group_id: str, 
                              metric_type: str = None) -> List[Dict[str, Any]]:
        """Get all metrics for a peer group"""
        conn = None
        try:
            conn = self.db_manager.engine.connect()
            
            if metric_type:
                query = text("""
                    SELECT 
                        peer_group_id, symbol, metric_type, sector, industry,
                        metrics_data, peer_symbols, calculation_date,
                        created_at, updated_at
                    FROM peer_metrics
                    WHERE peer_group_id = :peer_group_id 
                      AND metric_type = :metric_type
                    ORDER BY symbol
                """)
                params = {'peer_group_id': peer_group_id, 'metric_type': metric_type}
            else:
                query = text("""
                    SELECT 
                        peer_group_id, symbol, metric_type, sector, industry,
                        metrics_data, peer_symbols, calculation_date,
                        created_at, updated_at
                    FROM peer_metrics
                    WHERE peer_group_id = :peer_group_id
                    ORDER BY metric_type, symbol
                """)
                params = {'peer_group_id': peer_group_id}
            
            results = conn.execute(query, params).fetchall()
            
            return [
                {
                    'peer_group_id': row.peer_group_id,
                    'symbol': row.symbol,
                    'metric_type': row.metric_type,
                    'sector': row.sector,
                    'industry': row.industry,
                    'metrics_data': row.metrics_data,
                    'peer_symbols': row.peer_symbols,
                    'calculation_date': row.calculation_date.isoformat() if row.calculation_date else None,
                    'created_at': row.created_at.isoformat() if row.created_at else None,
                    'updated_at': row.updated_at.isoformat() if row.updated_at else None
                }
                for row in results
            ]
            
        except Exception as e:
            self.logger.error(f"Error getting peer group metrics: {e}")
            return []
        finally:
            if conn:
                conn.close()
    
    def get_symbol_metrics(self, symbol: str) -> List[Dict[str, Any]]:
        """Get all metrics for a specific symbol across all peer groups"""
        conn = None
        try:
            conn = self.db_manager.engine.connect()
            
            query = text("""
                SELECT 
                    peer_group_id, symbol, metric_type, sector, industry,
                    metrics_data, peer_symbols, calculation_date,
                    created_at, updated_at
                FROM peer_metrics
                WHERE symbol = :symbol
                ORDER BY peer_group_id, metric_type
            """)
            
            results = conn.execute(query, {'symbol': symbol}).fetchall()
            
            return [
                {
                    'peer_group_id': row.peer_group_id,
                    'symbol': row.symbol,
                    'metric_type': row.metric_type,
                    'sector': row.sector,
                    'industry': row.industry,
                    'metrics_data': row.metrics_data,
                    'peer_symbols': row.peer_symbols,
                    'calculation_date': row.calculation_date.isoformat() if row.calculation_date else None,
                    'created_at': row.created_at.isoformat() if row.created_at else None,
                    'updated_at': row.updated_at.isoformat() if row.updated_at else None
                }
                for row in results
            ]
            
        except Exception as e:
            self.logger.error(f"Error getting symbol metrics: {e}")
            return []
        finally:
            if conn:
                conn.close()
    
    def delete_peer_metrics(self, peer_group_id: str = None, symbol: str = None,
                           metric_type: str = None) -> int:
        """Delete peer metrics based on criteria"""
        conn = None
        try:
            conn = self.db_manager.engine.connect()
            
            conditions = []
            params = {}
            
            if peer_group_id:
                conditions.append("peer_group_id = :peer_group_id")
                params['peer_group_id'] = peer_group_id
            
            if symbol:
                conditions.append("symbol = :symbol")
                params['symbol'] = symbol
            
            if metric_type:
                conditions.append("metric_type = :metric_type")
                params['metric_type'] = metric_type
            
            if not conditions:
                self.logger.warning("No conditions specified for deletion")
                return 0
            
            where_clause = " AND ".join(conditions)
            query = text(f"DELETE FROM peer_metrics WHERE {where_clause}")
            
            with conn.begin():
                result = conn.execute(query, params)
                deleted_count = result.rowcount
            
            self.logger.info(f"Deleted {deleted_count} peer metrics records")
            return deleted_count
            
        except Exception as e:
            self.logger.error(f"Error deleting peer metrics: {e}")
            return 0
        finally:
            if conn:
                conn.close()
    
    def get_latest_calculation_date(self, peer_group_id: str = None, 
                                   symbol: str = None) -> Optional[date]:
        """Get the latest calculation date for given criteria"""
        conn = None
        try:
            conn = self.db_manager.engine.connect()
            
            conditions = []
            params = {}
            
            if peer_group_id:
                conditions.append("peer_group_id = :peer_group_id")
                params['peer_group_id'] = peer_group_id
            
            if symbol:
                conditions.append("symbol = :symbol")
                params['symbol'] = symbol
            
            where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
            
            query = text(f"""
                SELECT MAX(calculation_date) as latest_date
                FROM peer_metrics
                {where_clause}
            """)
            
            result = conn.execute(query, params).fetchone()
            
            return result.latest_date if result and result.latest_date else None
            
        except Exception as e:
            self.logger.error(f"Error getting latest calculation date: {e}")
            return None
        finally:
            if conn:
                conn.close()


# Singleton instance
_peer_metrics_dao = None


def get_peer_metrics_dao() -> PeerMetricsDAO:
    """Get singleton instance of PeerMetricsDAO"""
    global _peer_metrics_dao
    if _peer_metrics_dao is None:
        _peer_metrics_dao = PeerMetricsDAO()
    return _peer_metrics_dao