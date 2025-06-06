"""
Peer Group Analyzer for Russell 1000 stocks
Provides utilities for working with peer groups and relative valuation
"""

import json
import logging
from typing import Dict, List, Set, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class PeerGroupAnalyzer:
    """Analyzer for working with Russell 1000 peer groups"""
    
    def __init__(self, peer_groups_file: str = "data/russell_1000_peer_groups.json"):
        """Initialize with peer groups data file"""
        self.peer_groups_file = Path(peer_groups_file)
        self.peer_groups_data = self._load_peer_groups()
        self._build_ticker_index()
    
    def _load_peer_groups(self) -> Dict:
        """Load peer groups from JSON file"""
        try:
            with open(self.peer_groups_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Peer groups file not found: {self.peer_groups_file}")
            return {}
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in peer groups file: {self.peer_groups_file}")
            return {}
    
    def _build_ticker_index(self):
        """Build reverse index from ticker to sector/industry"""
        self.ticker_to_location = {}
        
        if not self.peer_groups_data:
            return
            
        for sector, industries in self.peer_groups_data.get('peer_groups', {}).items():
            for industry, data in industries.items():
                if isinstance(data, dict):
                    for cap_type in ['large_cap', 'mid_cap']:
                        if cap_type in data:
                            for ticker in data[cap_type]:
                                self.ticker_to_location[ticker] = {
                                    'sector': sector,
                                    'industry': industry,
                                    'cap_type': cap_type
                                }
    
    def get_peers(self, ticker: str, same_cap_only: bool = False) -> List[str]:
        """Get peer companies for a given ticker"""
        if ticker not in self.ticker_to_location:
            logger.warning(f"Ticker {ticker} not found in peer groups")
            return []
        
        location = self.ticker_to_location[ticker]
        sector = location['sector']
        industry = location['industry']
        cap_type = location['cap_type']
        
        # Get industry data
        industry_data = self.peer_groups_data['peer_groups'][sector][industry]
        
        peers = []
        if same_cap_only:
            peers = [t for t in industry_data.get(cap_type, []) if t != ticker]
        else:
            # Include both large and mid cap
            for cap in ['large_cap', 'mid_cap']:
                peers.extend([t for t in industry_data.get(cap, []) if t != ticker])
        
        return peers
    
    def get_sector_peers(self, ticker: str) -> List[str]:
        """Get all companies in the same sector"""
        if ticker not in self.ticker_to_location:
            return []
        
        sector = self.ticker_to_location[ticker]['sector']
        peers = []
        
        for industry, data in self.peer_groups_data['peer_groups'][sector].items():
            if isinstance(data, dict):
                for cap_type in ['large_cap', 'mid_cap']:
                    peers.extend(data.get(cap_type, []))
        
        return [p for p in peers if p != ticker]
    
    def get_industry_leaders(self, ticker: str) -> List[str]:
        """Get large cap leaders in the same industry"""
        if ticker not in self.ticker_to_location:
            return []
        
        location = self.ticker_to_location[ticker]
        sector = location['sector']
        industry = location['industry']
        
        industry_data = self.peer_groups_data['peer_groups'][sector][industry]
        return industry_data.get('large_cap', [])
    
    def get_ticker_info(self, ticker: str) -> Optional[Dict]:
        """Get sector, industry, and cap type for a ticker"""
        return self.ticker_to_location.get(ticker)
    
    def get_all_tickers_by_sector(self, sector: str) -> List[str]:
        """Get all tickers in a specific sector"""
        tickers = []
        
        if sector not in self.peer_groups_data.get('peer_groups', {}):
            return tickers
        
        for industry, data in self.peer_groups_data['peer_groups'][sector].items():
            if isinstance(data, dict):
                for cap_type in ['large_cap', 'mid_cap']:
                    tickers.extend(data.get(cap_type, []))
        
        return tickers
    
    def get_all_tickers_by_industry(self, sector: str, industry: str) -> List[str]:
        """Get all tickers in a specific industry"""
        tickers = []
        
        try:
            industry_data = self.peer_groups_data['peer_groups'][sector][industry]
            for cap_type in ['large_cap', 'mid_cap']:
                tickers.extend(industry_data.get(cap_type, []))
        except KeyError:
            logger.warning(f"Industry {industry} in sector {sector} not found")
        
        return tickers
    
    def get_comparison_group(self, ticker: str, max_peers: int = 10) -> List[str]:
        """Get a balanced comparison group for valuation analysis"""
        if ticker not in self.ticker_to_location:
            return []
        
        location = self.ticker_to_location[ticker]
        
        # Start with direct industry peers
        peers = self.get_peers(ticker, same_cap_only=False)
        
        # If we need more, add sector peers
        if len(peers) < max_peers:
            sector_peers = self.get_sector_peers(ticker)
            # Add peers from same cap type first
            same_cap = [p for p in sector_peers if p not in peers and 
                       self.ticker_to_location[p]['cap_type'] == location['cap_type']]
            peers.extend(same_cap[:max_peers - len(peers)])
        
        return peers[:max_peers]
    
    def get_sector_etf(self, sector: str) -> Optional[str]:
        """Get the SPDR sector ETF for a given sector"""
        return self.peer_groups_data.get('sector_mappings', {}).get(sector)
    
    def list_all_sectors(self) -> List[str]:
        """List all available sectors"""
        return list(self.peer_groups_data.get('peer_groups', {}).keys())
    
    def list_all_industries(self, sector: str) -> List[str]:
        """List all industries within a sector"""
        if sector not in self.peer_groups_data.get('peer_groups', {}):
            return []
        return list(self.peer_groups_data['peer_groups'][sector].keys())
    
    def get_industry_description(self, sector: str, industry: str) -> str:
        """Get description for an industry"""
        try:
            return self.peer_groups_data['peer_groups'][sector][industry].get('description', '')
        except KeyError:
            return ""
    
    def get_stats(self) -> Dict:
        """Get statistics about the peer groups"""
        stats = {
            'total_sectors': len(self.list_all_sectors()),
            'total_industries': 0,
            'total_companies': len(self.ticker_to_location),
            'companies_by_sector': {},
            'companies_by_cap': {'large_cap': 0, 'mid_cap': 0}
        }
        
        for sector in self.list_all_sectors():
            sector_count = len(self.get_all_tickers_by_sector(sector))
            stats['companies_by_sector'][sector] = sector_count
            stats['total_industries'] += len(self.list_all_industries(sector))
        
        for ticker, info in self.ticker_to_location.items():
            stats['companies_by_cap'][info['cap_type']] += 1
        
        return stats


# Example usage functions
def get_valuation_peers(ticker: str, analyzer: PeerGroupAnalyzer) -> Dict[str, List[str]]:
    """Get different peer groups for valuation analysis"""
    return {
        'direct_peers': analyzer.get_peers(ticker),
        'industry_leaders': analyzer.get_industry_leaders(ticker),
        'sector_peers': analyzer.get_sector_peers(ticker)[:20],  # Limit sector peers
        'comparison_group': analyzer.get_comparison_group(ticker)
    }


def print_peer_analysis(ticker: str, analyzer: PeerGroupAnalyzer):
    """Print a comprehensive peer analysis for a ticker"""
    info = analyzer.get_ticker_info(ticker)
    if not info:
        print(f"Ticker {ticker} not found in peer groups")
        return
    
    print(f"\n=== Peer Analysis for {ticker} ===")
    print(f"Sector: {info['sector']}")
    print(f"Industry: {info['industry']}")
    print(f"Market Cap: {info['cap_type'].replace('_', ' ').title()}")
    
    desc = analyzer.get_industry_description(info['sector'], info['industry'])
    if desc:
        print(f"Industry Description: {desc}")
    
    peers = get_valuation_peers(ticker, analyzer)
    
    print(f"\nDirect Industry Peers ({len(peers['direct_peers'])}): {', '.join(peers['direct_peers'])}")
    print(f"Industry Leaders: {', '.join(peers['industry_leaders'])}")
    print(f"Recommended Comparison Group: {', '.join(peers['comparison_group'])}")
    
    etf = analyzer.get_sector_etf(info['sector'])
    if etf:
        print(f"Sector ETF: {etf}")


if __name__ == "__main__":
    # Example usage
    analyzer = PeerGroupAnalyzer()
    
    # Print some example analyses
    example_tickers = ['AAPL', 'JPM', 'AMZN', 'PFE', 'XOM']
    
    for ticker in example_tickers:
        print_peer_analysis(ticker, analyzer)
    
    # Print overall statistics
    print("\n=== Peer Groups Statistics ===")
    stats = analyzer.get_stats()
    print(f"Total Sectors: {stats['total_sectors']}")
    print(f"Total Industries: {stats['total_industries']}")
    print(f"Total Companies: {stats['total_companies']}")
    print(f"Large Cap Companies: {stats['companies_by_cap']['large_cap']}")
    print(f"Mid Cap Companies: {stats['companies_by_cap']['mid_cap']}")
    
    print("\nCompanies by Sector:")
    for sector, count in stats['companies_by_sector'].items():
        print(f"  {sector}: {count}")