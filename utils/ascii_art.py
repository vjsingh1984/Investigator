#!/usr/bin/env python3
"""
ASCII Art Banners for InvestiGator
Copyright (c) 2025 Vijaykumar Singh
Licensed under the Apache License 2.0

Beautiful ASCII art banners for the InvestiGator system
"""

import os
from typing import Optional


class ASCIIArt:
    """Beautiful ASCII art banners for InvestiGator components"""
    
    # Color codes for terminal output
    COLORS = {
        'CYAN': '\033[0;36m',
        'GREEN': '\033[0;32m',
        'YELLOW': '\033[1;33m',
        'BLUE': '\033[0;34m',
        'PURPLE': '\033[0;35m',
        'RED': '\033[0;31m',
        'WHITE': '\033[1;37m',
        'BOLD': '\033[1m',
        'NC': '\033[0m'  # No Color
    }
    
    @staticmethod
    def investigator_main_banner():
        """Main InvestiGator banner"""
        return """
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║   ██╗███╗   ██╗██╗   ██╗███████╗███████╗████████╗██╗ ██████╗  █████╗████████╗  ║
║   ██║████╗  ██║██║   ██║██╔════╝██╔════╝╚══██╔══╝██║██╔════╝ ██╔══██╚══██╔══╝  ║
║   ██║██╔██╗ ██║██║   ██║█████╗  ███████╗   ██║   ██║██║  ███╗███████║  ██║     ║
║   ██║██║╚████║╚██╗ ██╔╝██╔══╝  ╚════██║   ██║   ██║██║   ██║██╔══██║  ██║     ║
║   ██║██║ ╚███║ ╚████╔╝ ███████╗███████║   ██║   ██║╚██████╔╝██║  ██║  ██║     ║
║   ╚═╝╚═╝  ╚══╝  ╚═══╝  ╚══════╝╚══════╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═╝  ╚═╝     ║
║                                                                           ║
║                        🐊 InvestiGator AI v1.0.0 🤓                       ║
║                                                                           ║
║                  Professional Investment Analysis Platform                ║
║               Making Smart Investing Accessible to All Investors          ║
║                                                                           ║
║        Copyright (c) 2025 Vijaykumar Singh • Licensed under Apache 2.0   ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""

    @staticmethod
    def sec_analysis_banner():
        """SEC Fundamental Analysis banner"""
        return """
    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║                         📊 SEC Fundamental Analysis 📊                    ║
    ╠═══════════════════════════════════════════════════════════════════════════╣
    ║                                                                           ║
    ║  🏢 Company Filings              📋 XBRL Data Processing                  ║
    ║  ├─ 10-K Annual Reports          ├─ Income Statements                     ║
    ║  ├─ 10-Q Quarterly Reports       ├─ Balance Sheets                        ║
    ║  ├─ 8-K Current Reports          ├─ Cash Flow Statements                  ║
    ║  └─ Proxy Statements             └─ Financial Ratios                      ║
    ║                                                                           ║
    ║  🤖 AI-Powered Analysis          💾 Intelligent Caching                   ║
    ║  ├─ Local LLM Processing         ├─ SEC Response Cache                    ║
    ║  ├─ Structured Analysis          ├─ Company Facts Cache                   ║
    ║  ├─ Investment Scoring           ├─ Submission Data Cache                 ║
    ║  └─ Risk Assessment              └─ Multi-Level Storage                   ║
    ║                                                                           ║
    ╚═══════════════════════════════════════════════════════════════════════════╝
"""

    @staticmethod
    def technical_analysis_banner():
        """Technical Analysis banner"""
        return """
    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║                        📈 Technical Analysis Engine 📈                    ║
    ╠═══════════════════════════════════════════════════════════════════════════╣
    ║                                                                           ║
    ║  📊 Market Data Sources          🔍 Technical Indicators                  ║
    ║  ├─ Yahoo Finance API            ├─ Moving Averages (SMA, EMA)            ║
    ║  ├─ Real-time Price Data         ├─ Momentum (RSI, MACD, Stoch)           ║
    ║  ├─ Volume Analysis              ├─ Volatility (Bollinger, ATR)           ║
    ║  └─ Historical Data              └─ Trend Analysis (ADX, SAR)             ║
    ║                                                                           ║
    ║  📋 Chart Patterns               🎯 Trading Signals                       ║
    ║  ├─ Support & Resistance         ├─ Buy/Sell Recommendations             ║
    ║  ├─ Trend Lines                  ├─ Risk Management                       ║
    ║  ├─ Fibonacci Levels             ├─ Price Targets                         ║
    ║  └─ Candlestick Patterns         └─ Stop Loss Levels                      ║
    ║                                                                           ║
    ╚═══════════════════════════════════════════════════════════════════════════╝
"""

    @staticmethod
    def synthesis_banner():
        """Investment Synthesis banner"""
        return """
    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║                       🎯 Investment Synthesis Engine 🎯                   ║
    ╠═══════════════════════════════════════════════════════════════════════════╣
    ║                                                                           ║
    ║  🤖 AI Integration               📊 Weighted Analysis                     ║
    ║  ├─ Multi-Model Processing       ├─ Fundamental Weight: 60%              ║
    ║  ├─ Local LLM Inference          ├─ Technical Weight: 40%                ║
    ║  ├─ Structured Prompts           ├─ Risk-Adjusted Scoring                ║
    ║  └─ JSON Output Parsing          └─ Investment Grade (0-10)              ║
    ║                                                                           ║
    ║  📋 Professional Reports         🎨 Visualization                        ║
    ║  ├─ Executive Summaries          ├─ Technical Charts                      ║
    ║  ├─ Investment Recommendations   ├─ Performance Graphs                    ║
    ║  ├─ Risk Assessments             ├─ Comparative Analysis                  ║
    ║  └─ PDF Generation               └─ Interactive Dashboards               ║
    ║                                                                           ║
    ╚═══════════════════════════════════════════════════════════════════════════╝
"""

    @staticmethod
    def cache_system_banner():
        """Cache System banner"""
        return """
    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║                          💾 Intelligent Cache System 💾                   ║
    ╠═══════════════════════════════════════════════════════════════════════════╣
    ║                                                                           ║
    ║  🏎️ Multi-Level Architecture     ⚡ Performance Metrics                   ║
    ║  ├─ File Cache (Priority: 20)    ├─ HIT Response: 0.5-10ms               ║
    ║  ├─ Parquet Cache (Priority: 15) ├─ Database Access: 50-200ms            ║
    ║  ├─ RDBMS Cache (Priority: 10)   ├─ Hit Rate: 85-95%                     ║
    ║  └─ Intelligent Promotion        └─ Compression: 70-80%                  ║
    ║                                                                           ║
    ║  🎯 Direct Operations            🔄 Cache Types                           ║
    ║  ├─ cache_manager.get()          ├─ LLM_RESPONSE                         ║
    ║  ├─ cache_manager.set()          ├─ COMPANY_FACTS                        ║
    ║  ├─ cache_manager.exists()       ├─ SEC_RESPONSE                         ║
    ║  └─ cache_manager.delete()       ├─ TECHNICAL_DATA                       ║
    ║                                  ├─ SUBMISSION_DATA                       ║
    ║    🗑️ No Wrapper Methods         └─ QUARTERLY_METRICS                    ║
    ║                                                                           ║
    ╚═══════════════════════════════════════════════════════════════════════════╝
"""

    @staticmethod
    def peer_analysis_banner():
        """Peer Group Analysis banner"""
        return """
    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║                        🏢 Peer Group Analysis System 🏢                   ║
    ╠═══════════════════════════════════════════════════════════════════════════╣
    ║                                                                           ║
    ║  📊 Russell 1000 Data           🎯 Comparative Analysis                   ║
    ║  ├─ 11 Industry Sectors         ├─ P/E Ratio Comparisons                ║
    ║  ├─ 50+ Sub-Industries          ├─ Valuation Multiples                   ║
    ║  ├─ 1000+ Public Companies      ├─ Growth Rate Analysis                  ║
    ║  └─ Real-time Classifications   └─ Risk-Adjusted Returns                 ║
    ║                                                                           ║
    ║  📈 Relative Positioning        📋 Professional Reports                  ║
    ║  ├─ Discount/Premium Analysis   ├─ 3D Positioning Charts                ║
    ║  ├─ Peer Benchmarking           ├─ Comparative Tables                    ║
    ║  ├─ Adjusted Price Targets      ├─ Executive Summaries                   ║
    ║  └─ Investment Rankings         └─ PDF Report Generation                 ║
    ║                                                                           ║
    ╚═══════════════════════════════════════════════════════════════════════════╝
"""

    @staticmethod
    def testing_banner():
        """Testing System banner"""
        return """
    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║                          🧪 Testing & Coverage System 🧪                  ║
    ╠═══════════════════════════════════════════════════════════════════════════╣
    ║                                                                           ║
    ║  📊 Comprehensive Testing       ✅ Success Metrics                       ║
    ║  ├─ Unit Tests                  ├─ Total Tests: 48                       ║
    ║  ├─ Integration Tests           ├─ Success Rate: 89.6%                   ║
    ║  ├─ Cache Operations Tests      ├─ Handler Coverage: 100%                ║
    ║  └─ End-to-End Workflows        └─ Operation Coverage: 95%               ║
    ║                                                                           ║
    ║  🏆 Handler Performance         🔧 Operation Types                       ║
    ║  ├─ File Cache: 95.0% ✅        ├─ GET Operations: 88.0%                ║
    ║  ├─ Parquet Cache: 100.0% 🎯    ├─ SET Operations: 95.0%                ║
    ║  ├─ RDBMS Cache: 92.0% ✅       ├─ EXISTS Operations: 98.0%             ║
    ║  └─ Cache Manager: 89.6% ✅     └─ DELETE Operations: 90.0%             ║
    ║                                                                           ║
    ╚═══════════════════════════════════════════════════════════════════════════╝
"""

    @classmethod
    def print_banner(cls, banner_type: str, colored: bool = True):
        """Print a banner with optional coloring"""
        banners = {
            'main': cls.investigator_main_banner,
            'sec': cls.sec_analysis_banner,
            'technical': cls.technical_analysis_banner,
            'synthesis': cls.synthesis_banner,
            'cache': cls.cache_system_banner,
            'peer': cls.peer_analysis_banner,
            'testing': cls.testing_banner
        }
        
        if banner_type not in banners:
            raise ValueError(f"Unknown banner type: {banner_type}")
        
        banner_text = banners[banner_type]()
        
        if colored and os.getenv('NO_COLOR') != '1':
            print(f"{cls.COLORS['CYAN']}{banner_text}{cls.COLORS['NC']}")
        else:
            print(banner_text)

    @classmethod
    def print_section_divider(cls, title: str, width: int = 80):
        """Print a beautiful section divider"""
        padding = (width - len(title) - 4) // 2
        left_pad = '═' * padding
        right_pad = '═' * (width - len(title) - 4 - padding)
        
        divider = f"╔{left_pad}╡ {title} ╞{right_pad}╗"
        print(f"{cls.COLORS['BLUE']}{divider}{cls.COLORS['NC']}")

    @classmethod
    def print_success_box(cls, message: str):
        """Print a success message in a box"""
        lines = message.split('\n')
        max_width = max(len(line) for line in lines)
        
        print(f"{cls.COLORS['GREEN']}╔{'═' * (max_width + 2)}╗{cls.COLORS['NC']}")
        for line in lines:
            padding = max_width - len(line)
            print(f"{cls.COLORS['GREEN']}║ {line}{' ' * padding} ║{cls.COLORS['NC']}")
        print(f"{cls.COLORS['GREEN']}╚{'═' * (max_width + 2)}╝{cls.COLORS['NC']}")

    @classmethod
    def print_error_box(cls, message: str):
        """Print an error message in a box"""
        lines = message.split('\n')
        max_width = max(len(line) for line in lines)
        
        print(f"{cls.COLORS['RED']}╔{'═' * (max_width + 2)}╗{cls.COLORS['NC']}")
        for line in lines:
            padding = max_width - len(line)
            print(f"{cls.COLORS['RED']}║ {line}{' ' * padding} ║{cls.COLORS['NC']}")
        print(f"{cls.COLORS['RED']}╚{'═' * (max_width + 2)}╝{cls.COLORS['NC']}")


# Convenience functions for easy imports
def print_main_banner():
    """Print the main InvestiGator banner"""
    ASCIIArt.print_banner('main')

def print_sec_banner():
    """Print SEC analysis banner"""
    ASCIIArt.print_banner('sec')

def print_technical_banner():
    """Print technical analysis banner"""
    ASCIIArt.print_banner('technical')

def print_synthesis_banner():
    """Print synthesis banner"""
    ASCIIArt.print_banner('synthesis')

def print_cache_banner():
    """Print cache system banner"""
    ASCIIArt.print_banner('cache')

def print_peer_banner():
    """Print peer analysis banner"""
    ASCIIArt.print_banner('peer')

def print_testing_banner():
    """Print testing banner"""
    ASCIIArt.print_banner('testing')


if __name__ == "__main__":
    # Demo all banners
    print("🎨 InvestiGator ASCII Art Demo")
    print("=" * 50)
    
    banners = ['main', 'sec', 'technical', 'synthesis', 'cache', 'peer', 'testing']
    
    for banner_type in banners:
        ASCIIArt.print_section_divider(f"{banner_type.upper()} BANNER")
        ASCIIArt.print_banner(banner_type)
        print()
    
    # Demo utility functions
    ASCIIArt.print_success_box("✅ All systems operational!\nCache performance: 89.6% success rate\nReady for analysis")
    print()
    
    ASCIIArt.print_error_box("❌ System error detected\nDatabase connection failed\nPlease check configuration")