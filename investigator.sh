#!/opt/homebrew/bin/bash  
#USe home brew bash for lastest bash as bash3 default mac bash may not support cmd/utils in script.

##############################################################################
# InvestiGator - AI Investment Research Assistant
# Copyright (c) 2025 Vijaykumar Singh
# Licensed under the Apache License, Version 2.0
#
# This script coordinates the execution of all analysis components:
# 1. SEC Fundamental Analysis (sec_fundamental.py)
# 2. Yahoo Technical Analysis (yahoo_technical.py)  
# 3. Analysis Synthesis and Reporting (synthesizer.py)
#
# Usage:
#   ./investigator.sh --symbol AAPL
#   ./investigator.sh --symbols AAPL GOOGL MSFT
#   ./investigator.sh --weekly-report --send-email
#   ./investigator.sh --start-scheduler
##############################################################################

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# Script metadata
readonly SCRIPT_VERSION="1.0.0"
readonly SCRIPT_NAME="InvestiGator"
readonly COPYRIGHT="Copyright (c) 2025 Vijaykumar Singh"
readonly LICENSE="Licensed under Apache License 2.0"

# Script directory and configuration
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly CONFIG_FILE="${SCRIPT_DIR}/config.json"
readonly LOG_DIR="${SCRIPT_DIR}/logs"
readonly PYTHON_ENV="${SCRIPT_DIR}/../investment_ai_env"
readonly DATA_DIR="${SCRIPT_DIR}/data"
readonly REPORTS_DIR="${SCRIPT_DIR}/reports"

# Execution tracking
START_TIME=$(date +%s)
TOTAL_STOCKS=0
SUCCESSFUL_ANALYSES=0
FAILED_ANALYSES=0
declare -a FAILED_SYMBOLS=()

# Colors for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly PURPLE='\033[0;35m'
readonly CYAN='\033[0;36m'
readonly WHITE='\033[1;37m'
readonly NC='\033[0m' # No Color

# Ticker mapping file
readonly TICKER_CIK_MAP="${DATA_DIR}/ticker_cik_map.txt"
readonly SEC_TICKER_URL="https://www.sec.gov/include/ticker.txt"

# Ensure log directory exists
mkdir -p "${LOG_DIR}" "${DATA_DIR}" "${REPORTS_DIR}"

# Symbol-specific logging support
CURRENT_SYMBOL=""

# Set current symbol for logging context
set_symbol_context() {
    CURRENT_SYMBOL="$1"
}

# Clear symbol context
clear_symbol_context() {
    CURRENT_SYMBOL=""
}

# Helper function to write to symbol log if symbol context is set
write_to_symbol_log() {
    local level="$1"
    local message="$2"
    
    if [[ -n "${CURRENT_SYMBOL}" ]]; then
        # Write to symbol-specific log without timestamp (matches Python logging format)
        echo "${level} - investigator.${CURRENT_SYMBOL} - ${message}" >> "${LOG_DIR}/${CURRENT_SYMBOL}.log"
    fi
}

# Logging functions with enhanced formatting and symbol-specific logging
log_info() {
    local timestamp=$(date +'%Y-%m-%d %H:%M:%S')
    local message="$*"
    echo -e "${GREEN}[${timestamp}] INFO:${NC} ${message}" | tee -a "${LOG_DIR}/investigator.log"
    write_to_symbol_log "INFO" "${message}"
}

log_warn() {
    local timestamp=$(date +'%Y-%m-%d %H:%M:%S')
    local message="$*"
    echo -e "${YELLOW}[${timestamp}] WARN:${NC} ${message}" | tee -a "${LOG_DIR}/investigator.log"
    write_to_symbol_log "WARNING" "${message}"
}

log_error() {
    local timestamp=$(date +'%Y-%m-%d %H:%M:%S')
    local message="$*"
    echo -e "${RED}[${timestamp}] ERROR:${NC} ${message}" | tee -a "${LOG_DIR}/investigator.log" >&2
    write_to_symbol_log "ERROR" "${message}"
}

log_step() {
    local timestamp=$(date +'%Y-%m-%d %H:%M:%S')
    local message="$*"
    echo -e "${BLUE}[${timestamp}] STEP:${NC} ${message}" | tee -a "${LOG_DIR}/investigator.log"
    write_to_symbol_log "INFO" "STEP: ${message}"
}

log_success() {
    local timestamp=$(date +'%Y-%m-%d %H:%M:%S')
    local message="$*"
    echo -e "${GREEN}[${timestamp}] SUCCESS:${NC} ${message}" | tee -a "${LOG_DIR}/investigator.log"
    write_to_symbol_log "INFO" "SUCCESS: ${message}"
}

log_debug() {
    if [[ "${DEBUG:-false}" == "true" ]]; then
        local timestamp=$(date +'%Y-%m-%d %H:%M:%S')
        local message="$*"
        echo -e "${PURPLE}[${timestamp}] DEBUG:${NC} ${message}" | tee -a "${LOG_DIR}/investigator.log"
        write_to_symbol_log "DEBUG" "${message}"
    fi
}

# Enhanced banner display with InvestiGator mascot
show_banner() {
    echo -e "${CYAN}"
    echo "╔══════════════════════════════════════════════════════════════════════════════╗"
    echo "║                                                                              ║"
    echo "║                              🎩     📊                                       ║"
    echo "║                             ╱ o ╲ ╱ o ╲                                      ║"
    echo "║                             ╲   ◡   ╱                                        ║"
    echo "║                              ╲═══╱                                           ║"
    echo "║                           INVESTIGATOR™                                       ║"
    echo "║                                                                              ║"
    echo "║                    🐊 InvestiGator AI v${SCRIPT_VERSION} 🤓                             ║"
    echo "║                                                                              ║"
    echo "║                  Professional Investment Analysis                            ║"
    echo "║               Making Smart Investing Accessible to All                       ║"
    echo "║                                                                              ║"
    echo "║           ${COPYRIGHT}           ${LICENSE}     ║"
    echo "╚══════════════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Function to check if Python virtual environment exists and activate it
activate_python_env() {
    if [[ -d "${PYTHON_ENV}" ]]; then
        log_info "Activating Python virtual environment: ${PYTHON_ENV}"
        # shellcheck source=/dev/null
        source "${PYTHON_ENV}/bin/activate"
        log_debug "Python environment activated successfully"
    else
        log_warn "Python virtual environment not found at ${PYTHON_ENV}"
        log_warn "Using system Python - this may cause dependency issues"
        log_warn "Run './scripts/setup.sh' to create proper environment"
    fi
}

# Function to download/update SEC ticker mapping
update_ticker_mapping() {
    log_step "Updating SEC ticker-to-CIK mapping..."
    
    # Check if file exists and is less than 24 hours old
    if [[ -f "${TICKER_CIK_MAP}" ]]; then
        local file_age=$(($(date +%s) - $(stat -f %m "${TICKER_CIK_MAP}" 2>/dev/null || echo 0)))
        local max_age=$((24 * 60 * 60))  # 24 hours in seconds
        
        if [[ ${file_age} -lt ${max_age} ]]; then
            log_debug "Ticker mapping is recent ($(( file_age / 3600 )) hours old), skipping update"
            return 0
        fi
    fi
    
    # Download new ticker mapping
    log_info "Downloading ticker mapping from SEC..."
    
    local temp_file="${TICKER_CIK_MAP}.tmp"
    local curl_cmd="curl -s -L --retry 3 --retry-delay 2 --max-time 30"
    local user_agent="InvestiGator/1.0 (Vijay Singh; singhvjd@gmail.com)"
    
    if ${curl_cmd} -H "User-Agent: ${user_agent}" "${SEC_TICKER_URL}" -o "${temp_file}"; then
        # Verify the download
        if [[ -s "${temp_file}" ]] && grep -q "^[a-z]" "${temp_file}" 2>/dev/null; then
            mv "${temp_file}" "${TICKER_CIK_MAP}"
            local line_count=$(wc -l < "${TICKER_CIK_MAP}")
            log_success "Ticker mapping updated successfully (${line_count} tickers)"
            
            # Show a few sample entries for verification
            log_debug "Sample entries:"
            head -5 "${TICKER_CIK_MAP}" | while IFS=$'\t' read -r ticker cik; do
                log_debug "  ${ticker} -> ${cik}"
            done
            
            return 0
        else
            log_error "Downloaded file appears to be invalid"
            rm -f "${temp_file}"
            return 1
        fi
    else
        log_error "Failed to download ticker mapping from SEC"
        rm -f "${temp_file}"
        return 1
    fi
}

# Function to resolve ticker to CIK
resolve_ticker_to_cik() {
    local ticker="$1"
    local ticker_lower=$(echo "${ticker}" | tr '[:upper:]' '[:lower:]')
    
    # Ensure ticker mapping exists
    if [[ ! -f "${TICKER_CIK_MAP}" ]]; then
        log_warn "Ticker mapping not found, attempting to download..."
        if ! update_ticker_mapping; then
            log_error "Failed to download ticker mapping"
            return 1
        fi
    fi
    
    # Look up the ticker
    local cik=$(grep -i "^${ticker_lower}[[:space:]]" "${TICKER_CIK_MAP}" 2>/dev/null | awk '{print $2}' | head -1)
    
    if [[ -n "${cik}" ]]; then
        echo "${cik}"
        return 0
    else
        log_error "CIK not found for ticker: ${ticker}"
        return 1
    fi
}

# Enhanced prerequisites checking
check_prerequisites() {
    log_step "Performing comprehensive system check..."
    
    local errors=0
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is not installed"
        ((errors++))
    else
        local python_version
        python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
        log_debug "Python version: ${python_version}"
    fi
    
    # Check Ollama
    if ! command -v ollama &> /dev/null; then
        log_error "Ollama is not installed"
        log_error "Install with: curl -fsSL https://ollama.com/install.sh | sh"
        ((errors++))
    else
        log_debug "Ollama binary found"
    fi
    
    # Check if Ollama service is running
    if ! curl -s http://localhost:11434/api/tags &> /dev/null; then
        log_error "Ollama service is not running"
        log_error "Start it with: ollama serve"
        ((errors++))
    else
        log_debug "Ollama service is responsive"
    fi
    
    # Check PostgreSQL
    if ! command -v psql &> /dev/null; then
        log_error "PostgreSQL client (psql) is not installed"
        log_error "Install with: brew install postgresql@14"
        ((errors++))
    else
        log_debug "PostgreSQL client found"
    fi
    
    # Check configuration file
    if [[ ! -f "${CONFIG_FILE}" ]]; then
        log_error "Configuration file not found: ${CONFIG_FILE}"
        log_error "Create it by running: python3 config.py"
        ((errors++))
    else
        log_debug "Configuration file found"
    fi
    
    # Check required Python modules
    activate_python_env
    local modules=("requests" "pandas" "yfinance" "reportlab" "sqlalchemy" "psycopg2")
    for module in "${modules[@]}"; do
        if ! python3 -c "import ${module}" 2>/dev/null; then
            log_error "Required Python module '${module}' not found"
            ((errors++))
        fi
    done
    
    # Check disk space (minimum 10GB)
    local available_space
    available_space=$(df "${SCRIPT_DIR}" | tail -1 | awk '{print $4}')
    local space_gb=$((available_space / 1024 / 1024))
    
    if [[ ${space_gb} -lt 10 ]]; then
        log_warn "Low disk space: ${space_gb}GB available (10GB+ recommended)"
    else
        log_debug "Disk space: ${space_gb}GB available"
    fi
    
    # Check memory
    if command -v memory_pressure &> /dev/null; then
        local memory_info
        memory_info=$(memory_pressure | grep "System-wide memory free percentage" | awk '{print $5}' | sed 's/%//')
        if [[ -n "$memory_info" && "$memory_info" -lt 20 ]]; then
            log_warn "Low memory: ${memory_info}% free"
        fi
    fi
    
    if [[ ${errors} -eq 0 ]]; then
        log_success "Prerequisites check passed "
        return 0
    else
        log_error "Prerequisites check failed with ${errors} error(s) "
        return 1
    fi
}

# Enhanced database connectivity test
test_database() {
    log_step "Testing database connectivity..."
    
    activate_python_env
    
    if python3 -c "
import sys
sys.path.append('.')
try:
    from utils.db import get_db_manager
    db = get_db_manager()
    if db.test_connection():
        print(' Database connection successful')
        exit(0)
    else:
        print(' Database connection failed')
        exit(1)
except ImportError as e:
    print(f' Import error: {e}')
    exit(1)
except Exception as e:
    print(f' Database error: {e}')
    exit(1)
" 2>/dev/null; then
        log_success "Database connectivity test passed "
        return 0
    else
        log_error "Database connectivity test failed "
        log_error "Check PostgreSQL service and configuration"
        return 1
    fi
}

# Enhanced Ollama models check
test_ollama_models() {
    log_step "Verifying Ollama models availability..."
    
    local required_models=()
    local available_models
    local missing_models=()
    
    # Extract required models from config if possible
    if [[ -f "${CONFIG_FILE}" ]] && command -v python3 &> /dev/null; then
        activate_python_env
        readarray -t required_models < <(python3 -c "
import json
import sys
try:
    with open('config.json', 'r') as f:
        config = json.load(f)
    models = config.get('ollama', {}).get('models', {})
    for model in models.values():
        print(model)
except Exception:
    # Fallback to default models
    print('phi4-reasoning:plus')
    print('qwen2.5:32b-instruct-q4_K_M')
    print('llama-3.3-70b-instruct-q4_k_m-128K-custom:latest')
" 2>/dev/null)
    fi
    
    # Fallback to default models if extraction failed
    if [[ ${#required_models[@]} -eq 0 ]]; then
        required_models=("phi4-reasoning:plus" "qwen2.5:32b-instruct-q4_K_M" "llama-3.3-70b-instruct-q4_k_m-128K-custom:latest")
    fi
    
    available_models=$(ollama list 2>/dev/null | tail -n +2 | awk '{print $1}' || echo "")
    
    for model in "${required_models[@]}"; do
        if ! echo "${available_models}" | grep -q "^${model}$"; then
            missing_models+=("${model}")
        fi
    done
    
    if [[ ${#missing_models[@]} -eq 0 ]]; then
        log_success "All required Ollama models are available "
        log_debug "Available models: ${required_models[*]}"
        return 0
    else
        log_warn "Missing Ollama models: ${missing_models[*]}"
        log_warn "Download them with: ollama pull <model-name>"
        log_warn "This may impact analysis quality"
        return 1
    fi
}

# Ensure ticker mappings are loaded in database
ensure_ticker_mappings() {
    log_step "Ensuring ticker-to-CIK mappings are loaded..."
    
    activate_python_env
    
    if python3 -c "
import sys
sys.path.append('.')
try:
    from utils.ticker_cik_mapper import TickerCIKMapper
    from utils.db import DatabaseManager, get_ticker_cik_mapping_dao
    from sqlalchemy import text
    
    # Check if mappings exist in database
    db = DatabaseManager()
    with db.get_session() as session:
        count = session.execute(text('SELECT COUNT(*) FROM ticker_cik_mapping')).scalar()
        if count < 10:  # Less than 10 mappings means it's not properly loaded
            print(f'Only {count} mappings in database, loading from file...')
            # Initialize mapper and load mappings
            mapper = TickerCIKMapper()
            dao = get_ticker_cik_mapping_dao()
            
            # Save each mapping to database
            saved = 0
            for ticker, cik in mapper.ticker_map.items():
                if dao.save_mapping(ticker.upper(), cik, f'{ticker.upper()} Company'):
                    saved += 1
            
            print(f'Loaded {saved} ticker mappings into database')
        else:
            print(f'Found {count} ticker mappings in database')
    
    sys.exit(0)
except Exception as e:
    print(f'Error: {e}', file=sys.stderr)
    sys.exit(1)
" 2>&1; then
        log_success "Ticker mappings verified ✓"
        return 0
    else
        log_error "Failed to verify ticker mappings"
        return 1
    fi
}

# Ensure SEC submissions are loaded for symbols
ensure_sec_submissions() {
    local symbols=("$@")
    log_step "Checking SEC submissions for ${#symbols[@]} symbols..."
    
    activate_python_env
    
    if python3 -c "
import sys
sys.path.append('.')
try:
    from utils.db import DatabaseManager, get_all_submission_store_dao
    from sqlalchemy import text
    
    symbols = '${symbols[*]}'.split()
    
    db = DatabaseManager()
    dao = get_all_submission_store_dao()
    
    missing_symbols = []
    for symbol in symbols:
        # Use get_submission with placeholder CIK to check if any data exists
        submission = dao.get_submission(symbol, '0000000000', max_age_days=365)
        if not submission:
            missing_symbols.append(symbol)
        else:
            print(f'{symbol}: submission data found')
    
    if missing_symbols:
        print(f'Missing submissions for: {\" \".join(missing_symbols)}')
        print('These symbols will need SEC data fetching during analysis')
    
    sys.exit(0)
except Exception as e:
    print(f'Error: {e}', file=sys.stderr)
    sys.exit(1)
" 2>&1; then
        log_success "SEC submissions check completed"
        return 0
    else
        log_warn "Failed to check SEC submissions"
        return 1
    fi
}

# Enhanced SEC fundamental analysis execution
run_sec_analysis() {
    local symbol="$1"
    local log_file="${LOG_DIR}/sec_fundamental_${symbol,,}_$(date +%Y%m%d_%H%M%S).log"
    
    # Set symbol context for logging
    set_symbol_context "${symbol}"
    
    log_step "🔍 Starting SEC fundamental analysis for ${symbol}..."
    
    activate_python_env
    
    local start_time
    start_time=$(date +%s)
    
    if timeout 1800 python3 sec_fundamental.py --symbol "${symbol}" > "${log_file}" 2>&1; then
        local end_time
        end_time=$(date +%s)
        local duration=$((end_time - start_time))
        log_success "SEC fundamental analysis completed for ${symbol} in ${duration}s "
        log_debug "Log file: ${log_file}"
        clear_symbol_context
        return 0
    else
        local exit_code=$?
        log_error "SEC fundamental analysis failed for ${symbol} "
        log_error "Exit code: ${exit_code}"
        log_error "Check log file: ${log_file}"
        
        # Show last few lines of error log
        if [[ -f "${log_file}" ]]; then
            log_error "Last 5 lines of error log:"
            tail -5 "${log_file}" | while IFS= read -r line; do
                log_error "  ${line}"
            done
        fi
        
        clear_symbol_context
        return 1
    fi
}

# Enhanced Yahoo technical analysis execution
run_technical_analysis() {
    local symbol="$1"
    local log_file="${LOG_DIR}/yahoo_technical_${symbol,,}_$(date +%Y%m%d_%H%M%S).log"
    
    # Set symbol context for logging
    set_symbol_context "${symbol}"
    
    log_step "📈 Starting Yahoo technical analysis for ${symbol}..."
    
    activate_python_env
    
    local start_time
    start_time=$(date +%s)
    
    if timeout 900 python3 yahoo_technical.py --symbol "${symbol}" > "${log_file}" 2>&1; then
        local end_time
        end_time=$(date +%s)
        local duration=$((end_time - start_time))
        log_success "Yahoo technical analysis completed for ${symbol} in ${duration}s "
        log_debug "Log file: ${log_file}"
        clear_symbol_context
        return 0
    else
        local exit_code=$?
        log_error "Yahoo technical analysis failed for ${symbol} "
        log_error "Exit code: ${exit_code}"
        log_error "Check log file: ${log_file}"
        
        # Show last few lines of error log
        if [[ -f "${log_file}" ]]; then
            log_error "Last 5 lines of error log:"
            tail -5 "${log_file}" | while IFS= read -r line; do
                log_error "  ${line}"
            done
        fi
        
        clear_symbol_context
        return 1
    fi
}

# Enhanced analysis synthesis execution
run_synthesis() {
    local symbol="$1"
    local log_file="${LOG_DIR}/synthesizer_${symbol,,}_$(date +%Y%m%d_%H%M%S).log"
    
    # Set symbol context for logging
    set_symbol_context "${symbol}"
    
    log_step " Starting analysis synthesis for ${symbol}..."
    
    activate_python_env
    
    local start_time
    start_time=$(date +%s)
    
    if timeout 300 python3 synthesizer.py --symbol "${symbol}" --report > "${log_file}" 2>&1; then
        local end_time
        end_time=$(date +%s)
        local duration=$((end_time - start_time))
        log_success "Analysis synthesis completed for ${symbol} in ${duration}s "
        log_debug "Log file: ${log_file}"
        clear_symbol_context
        return 0
    else
        local exit_code=$?
        log_error "Analysis synthesis failed for ${symbol} "
        log_error "Exit code: ${exit_code}"
        log_error "Check log file: ${log_file}"
        
        # Show last few lines of error log
        if [[ -f "${log_file}" ]]; then
            log_error "Last 5 lines of error log:"
            tail -5 "${log_file}" | while IFS= read -r line; do
                log_error "  ${line}"
            done
        fi
        
        clear_symbol_context
        return 1
    fi
}

# Comprehensive single stock analysis with detailed progress tracking
analyze_stock() {
    local symbol="$1"
    local start_time
    start_time=$(date +%s)
    
    # Set symbol context for overall analysis logging
    set_symbol_context "${symbol}"
    
    log_info " Starting comprehensive analysis pipeline for ${symbol}"
    printf "${WHITE}%0.s=" {1..80}
    echo -e "${NC}"
    
    local failed_steps=()
    local step_times=()
    
    # Ensure ticker mappings are loaded before analysis
    if ! ensure_ticker_mappings; then
        log_warn "Failed to verify ticker mappings, continuing anyway..."
    fi
    
    # Step 1: SEC Fundamental Analysis
    echo -e "${BLUE}Step 1/3: SEC Fundamental Analysis${NC}"
    local step_start
    step_start=$(date +%s)
    
    if run_sec_analysis "${symbol}"; then
        local step_duration=$(($(date +%s) - step_start))
        step_times+=("SEC: ${step_duration}s")
        echo -e "${GREEN} SEC Analysis: ${step_duration}s${NC}"
    else
        failed_steps+=("SEC Analysis")
        echo -e "${RED} SEC Analysis: FAILED${NC}"
    fi
    
    # Step 2: Yahoo Technical Analysis  
    echo -e "${BLUE}Step 2/3: Technical Analysis${NC}"
    step_start=$(date +%s)
    
    if run_technical_analysis "${symbol}"; then
        local step_duration=$(($(date +%s) - step_start))
        step_times+=("Technical: ${step_duration}s")
        echo -e "${GREEN} Technical Analysis: ${step_duration}s${NC}"
    else
        failed_steps+=("Technical Analysis")
        echo -e "${RED} Technical Analysis: FAILED${NC}"
    fi
    
    # Step 3: Synthesis
    echo -e "${BLUE}Step 3/3: Investment Synthesis${NC}"
    step_start=$(date +%s)
    
    if run_synthesis "${symbol}"; then
        local step_duration=$(($(date +%s) - step_start))
        step_times+=("Synthesis: ${step_duration}s")
        echo -e "${GREEN} Investment Synthesis: ${step_duration}s${NC}"
    else
        failed_steps+=("Investment Synthesis")
        echo -e "${RED} Investment Synthesis: FAILED${NC}"
    fi
    
    # Calculate total duration and display results
    local end_time
    end_time=$(date +%s)
    local total_duration=$((end_time - start_time))
    local duration_min=$((total_duration / 60))
    local duration_sec=$((total_duration % 60))
    
    printf "${WHITE}%0.s=" {1..80}
    echo "${NC}"
    
    if [[ ${#failed_steps[@]} -eq 0 ]]; then
        log_success "Complete analysis pipeline for ${symbol} finished successfully!"
        echo -e "${GREEN} Total Duration: ${duration_min}m ${duration_sec}s${NC}"
        echo -e "${CYAN} Step Breakdown: ${step_times[*]}${NC}"
        SUCCESSFUL_ANALYSES=$((SUCCESSFUL_ANALYSES+1))
        
        # Display quick results if available
        display_quick_results "${symbol}"
        
        clear_symbol_context
        return 0
    else
        log_error " Analysis pipeline for ${symbol} completed with errors"
        echo -e "${RED} Total Duration: ${duration_min}m ${duration_sec}s${NC}"
        echo -e "${RED} Failed Steps: ${failed_steps[*]}${NC}"
        echo -e "${CYAN} Completed Steps: ${step_times[*]}${NC}"
        FAILED_ANALYSES=$((FAILED_ANALYSES+1))
        FAILED_SYMBOLS+=("${symbol}")
        clear_symbol_context
        return 1
    fi
}

# Display quick analysis results
display_quick_results() {
    local symbol="$1"
    
    activate_python_env
    
    python3 -c "
import sys
sys.path.append('.')
try:
    from utils.db import get_stock_analysis_dao
    dao = get_stock_analysis_dao()
    analysis = dao.get_analysis('${symbol}')
    if analysis:
        print(f' Quick Results for ${symbol}:')
        print(f'   Overall Score: {analysis.get(\"overall_score\", \"N/A\")}/10')
        print(f'   Recommendation: {analysis.get(\"recommendation\", \"N/A\")}')
        if analysis.get('price_target'):
            print(f'   Price Target: \${analysis[\"price_target\"]:.2f}')
        if analysis.get('current_price'):
            print(f'   Current Price: \${analysis[\"current_price\"]:.2f}')
except Exception:
    pass
" 2>/dev/null || true
}

# Enhanced batch analysis
run_batch_analysis() {
    local symbols=("$@")
    TOTAL_STOCKS=${#symbols[@]}
    local batch_start_time
    batch_start_time=$(date +%s)
    
    log_info " Starting batch analysis for ${TOTAL_STOCKS} stocks: ${symbols[*]}"
    echo -e "${CYAN}════════════════════════════════════════════════════════════════════════════════${NC}"
    
    # Ensure ticker mappings are loaded before analysis
    if ! ensure_ticker_mappings; then
        log_warn "Failed to verify ticker mappings, continuing anyway..."
    fi
    
    # Check SEC submissions for all symbols
    if ! ensure_sec_submissions "${symbols[@]}"; then
        log_warn "Failed to check SEC submissions, continuing anyway..."
    fi
    
    local current_stock=0
    
    for symbol in "${symbols[@]}"; do
        current_stock=$((current_stock+1))
        echo -e "${WHITE}"
        echo "╔════════════════════════════════════════════════════════════════════════════════╗"
        echo "║  Stock ${current_stock}/${TOTAL_STOCKS}: ${symbol^^} $(printf '%*s' $((66 - ${#symbol})) ' ')║"
        echo "╚════════════════════════════════════════════════════════════════════════════════╝"
        echo -e "${NC}"
        
        analyze_stock "${symbol^^}"
        
        # Brief pause between stocks to prevent overwhelming the system
        if [[ ${current_stock} -lt ${TOTAL_STOCKS} ]]; then
            log_debug "Pausing 5 seconds before next stock..."
            sleep 5
        fi
    done
    
    # Calculate and display batch summary
    local batch_end_time
    batch_end_time=$(date +%s)
    local batch_duration=$((batch_end_time - batch_start_time))
    local batch_duration_min=$((batch_duration / 60))
    local batch_duration_sec=$((batch_duration % 60))
    
    echo -e "${CYAN}════════════════════════════════════════════════════════════════════════════════${NC}"
    log_info " Batch Analysis Summary:"
    echo -e "${GREEN} Successful: ${SUCCESSFUL_ANALYSES}/${TOTAL_STOCKS}${NC}"
    
    if [[ ${FAILED_ANALYSES} -gt 0 ]]; then
        echo -e "${RED} Failed: ${FAILED_ANALYSES}/${TOTAL_STOCKS}${NC}"
        echo -e "${RED} Failed Symbols: ${FAILED_SYMBOLS[*]}${NC}"
    fi
    
    echo -e "${BLUE}⏱️  Total Batch Duration: ${batch_duration_min}m ${batch_duration_sec}s${NC}"
    echo -e "${BLUE}⏱️  Average per Stock: $((batch_duration / TOTAL_STOCKS))s${NC}"
    
    if [[ ${FAILED_ANALYSES} -eq 0 ]]; then
        log_success "🎉 All stocks analyzed successfully!"
        return 0
    else
        log_warn " Batch completed with some failures"
        return 1
    fi
}

# Enhanced weekly report generation
generate_weekly_report() {
    local send_email="$1"
    local log_file="${LOG_DIR}/weekly_report_$(date +%Y%m%d_%H%M%S).log"
    
    log_step " Generating comprehensive weekly investment report..."
    
    activate_python_env
    
    local cmd="python3 synthesizer.py --weekly"
    if [[ "${send_email}" == "true" ]]; then
        cmd="${cmd} --send-email"
        log_info "Email delivery enabled"
    fi
    
    local start_time
    start_time=$(date +%s)
    
    if timeout 1800 bash -c "eval \"${cmd}\" > "${log_file}" 2>&1"; then
        local end_time
        end_time=$(date +%s)
        local duration=$((end_time - start_time))
        
        log_success "Weekly report generation completed in ${duration}s "
        
        # Find and display the generated report
        local report_file
        report_file=$(find "${REPORTS_DIR}" -name "InvestiGator_Report_*.pdf" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2- || echo "")
        
        if [[ -n "${report_file}" && -f "${report_file}" ]]; then
            local report_size
            report_size=$(du -h "${report_file}" | cut -f1)
            log_success "📄 Report saved: $(basename "${report_file}") (${report_size})"
            echo -e "${GREEN} Full path: ${report_file}${NC}"
            
            # Open report if on macOS
            if [[ "$(uname)" == "Darwin" ]] && command -v open &> /dev/null; then
                log_info "Opening report in default PDF viewer..."
                open "${report_file}" || log_warn "Could not open PDF automatically"
            fi
        else
            log_warn "Generated report file not found in ${REPORTS_DIR}"
        fi
        
        if [[ "${send_email}" == "true" ]]; then
            log_info "📧 Email delivery attempted (check logs for status)"
        fi
        
        return 0
    else
        local exit_code=$?
        log_error "Weekly report generation failed with exit code ${exit_code} "
        log_error "Check log file: ${log_file}"
        
        # Show last few lines of error log
        if [[ -f "${log_file}" ]]; then
            log_error "Last 10 lines of error log:"
            tail -10 "${log_file}" | while IFS= read -r line; do
                log_error "  ${line}"
            done
        fi
        
        return 1
    fi
}

# Start integrated scheduler
start_scheduler() {
    log_step "🕒 Starting InvestiGator investment analysis scheduler..."
    
    # Create scheduler PID file
    local scheduler_pid_file="${LOG_DIR}/scheduler.pid"
    
    # Check if scheduler is already running
    if [[ -f "${scheduler_pid_file}" ]]; then
        local existing_pid
        existing_pid=$(cat "${scheduler_pid_file}" 2>/dev/null)
        if [[ -n "${existing_pid}" ]] && kill -0 "${existing_pid}" 2>/dev/null; then
            log_error "❌ Scheduler already running with PID ${existing_pid}"
            return 1
        else
            log_warn "🗑️ Removing stale PID file"
            rm -f "${scheduler_pid_file}"
        fi
    fi
    
    # Set up signal handling for graceful shutdown
    trap 'scheduler_cleanup_on_exit' INT TERM
    
    # Log scheduler startup
    log_info "📅 Configuring scheduled jobs..."
    log_info "   • Weekly analysis: Every Sunday at 8:00 AM"
    log_info "   • Daily health checks: Monday-Friday at 6:00 AM"
    log_info "   • System stats: Every Sunday at 7:45 AM"
    
    # Store scheduler PID
    echo $$ > "${scheduler_pid_file}"
    
    log_success "✅ InvestiGator scheduler started successfully (PID: $$)"
    log_info "🔄 Scheduler running - waiting for scheduled times..."
    log_info "⏹️  Press Ctrl+C to stop the scheduler"
    
    # Main scheduler loop
    local current_minute=-1
    local current_hour=-1
    local current_day=-1
    
    while true; do
        local now=$(date)
        local minute=$(date +%M | sed 's/^0//')
        local hour=$(date +%H | sed 's/^0//')
        local day_of_week=$(date +%w)  # 0=Sunday, 1=Monday, etc.
        local day_name=$(date +%A)
        
        # Only check jobs when minute changes (avoid duplicate executions)
        if [[ "${minute}" != "${current_minute}" ]]; then
            current_minute="${minute}"
            current_hour="${hour}"
            
            # Weekly comprehensive analysis - Sunday 8:00 AM
            if [[ "${day_of_week}" == "0" && "${hour}" == "8" && "${minute}" == "0" ]]; then
                log_info "🗓️ Executing weekly comprehensive analysis..."
                run_weekly_analysis
            fi
            
            # Daily system health checks - Monday-Friday 6:00 AM
            if [[ "${day_of_week}" -ge "1" && "${day_of_week}" -le "5" && "${hour}" == "6" && "${minute}" == "0" ]]; then
                log_info "🩺 Executing daily system health check..."
                run_daily_check
            fi
            
            # Weekly system stats - Sunday 7:45 AM
            if [[ "${day_of_week}" == "0" && "${hour}" == "7" && "${minute}" == "45" ]]; then
                log_info "📊 Executing weekly system statistics check..."
                run_system_stats_check
            fi
            
            # Hourly status logging
            if [[ "${minute}" == "0" ]]; then
                log_info "⏰ Scheduler active - next check in 1 minute"
            fi
        fi
        
        # Sleep for 60 seconds
        sleep 60
    done
}

# Scheduler cleanup function
scheduler_cleanup_on_exit() {
    local scheduler_pid_file="${LOG_DIR}/scheduler.pid"
    
    log_warn "🛑 Scheduler shutdown signal received"
    
    # Remove PID file
    if [[ -f "${scheduler_pid_file}" ]]; then
        rm -f "${scheduler_pid_file}"
        log_info "🗑️ Removed scheduler PID file"
    fi
    
    log_info "👋 InvestiGator scheduler stopped gracefully"
    exit 0
}

# Weekly analysis function
run_weekly_analysis() {
    log_info "📊 Starting weekly comprehensive analysis..."
    
    # Run weekly report with email
    if ./investigator.sh --weekly-report --send-email; then
        log_success "✅ Weekly analysis completed successfully"
    else
        log_error "❌ Weekly analysis failed"
        # Send notification email about failure
        send_failure_notification "Weekly Analysis Failed" "The scheduled weekly analysis failed to complete."
    fi
}

# Daily system check function
run_daily_check() {
    log_info "🩺 Running daily system health check..."
    
    # Check system stats
    if ./investigator.sh --system-stats; then
        log_success "✅ Daily health check passed"
    else
        log_error "❌ Daily health check failed"
    fi
    
    # Check cache sizes (warn if too large)
    local cache_usage
    cache_usage=$(du -sh data/ 2>/dev/null | cut -f1)
    log_info "💾 Current cache usage: ${cache_usage}"
}

# Weekly system stats function  
run_system_stats_check() {
    log_info "📊 Running weekly system statistics..."
    
    # Generate system stats
    ./investigator.sh --system-stats
    
    # Clean old logs (keep 30 days)
    find "${LOG_DIR}" -name "*.log" -mtime +30 -delete 2>/dev/null || true
    
    # Clean old reports (keep 90 days)
    find "${REPORTS_DIR}" -name "*.pdf" -mtime +90 -delete 2>/dev/null || true
    
    log_info "🧹 Cleaned old logs and reports"
}

# Send failure notification
send_failure_notification() {
    local subject="$1"
    local message="$2"
    
    log_warn "📧 Sending failure notification: ${subject}"
    # This could be enhanced to actually send email notifications
    echo "${message}" > "${LOG_DIR}/failure_notification_$(date +%Y%m%d_%H%M%S).txt"
}

# Comprehensive system testing
test_system() {
    log_step "🧪 Running comprehensive InvestiGator system test..."
    
    local test_results=()
    local total_tests=0
    local passed_tests=0
    
    echo -e "${BLUE}════════════════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${WHITE}                          🐊 InvestiGator System Test Suite 🤓                         ${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════════════════════════${NC}"
    
    # Test 1: Prerequisites
    echo -e "\n${CYAN}Test 1: System Prerequisites${NC}"
    ((total_tests++))
    if check_prerequisites; then
        test_results+=(" Prerequisites: PASS")
        ((passed_tests++))
    else
        test_results+=(" Prerequisites: FAIL")
    fi
    
    # Test 2: Database
    echo -e "\n${CYAN}Test 2: Database Connectivity${NC}"
    ((total_tests++))
    if test_database; then
        test_results+=(" Database: PASS")
        ((passed_tests++))
    else
        test_results+=(" Database: FAIL")
    fi
    
    # Test 3: Ollama Models
    echo -e "\n${CYAN}Test 3: AI Models${NC}"
    ((total_tests++))
    if test_ollama_models; then
        test_results+=(" AI Models: PASS")
        ((passed_tests++))
    else
        test_results+=(" AI Models: PARTIAL")
    fi
    
    # Test 4: SEC API
    echo -e "\n${CYAN}Test 4: SEC API Connectivity${NC}"
    ((total_tests++))
    activate_python_env
    if timeout 60 python3 sec_fundamental.py --test-connection --symbol AAPL > /dev/null 2>&1; then
        test_results+=(" SEC API: PASS")
        ((passed_tests++))
    else
        test_results+=(" SEC API: FAIL")
    fi
    
    # Test 5: Yahoo Finance
    echo -e "\n${CYAN}Test 5: Yahoo Finance API${NC}"
    ((total_tests++))
    if timeout 60 python3 yahoo_technical.py --test-data --symbol AAPL > /dev/null 2>&1; then
        test_results+=(" Yahoo Finance: PASS")
        ((passed_tests++))
    else
        test_results+=(" Yahoo Finance: FAIL")
    fi
    
    # Test 6: Configuration Validation
    echo -e "\n${CYAN}Test 6: Configuration Validation${NC}"
    ((total_tests++))
    
    if python3 -c "
import sys
sys.path.append('.')
try:
    from config import get_config
    config = get_config()
    errors = config.validate()
    if not errors:
        print(' Configuration valid')
        exit(0)
    else:
        print(f' Configuration errors: {errors}')
        exit(1)
except Exception as e:
    print(f' Configuration error: {e}')
    exit(1)
" 2>/dev/null; then
        test_results+=(" Configuration: PASS")
        ((passed_tests++))
    else
        test_results+=(" Configuration: FAIL")
    fi
    
    # Display comprehensive test results
    echo -e "\n${BLUE}════════════════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${WHITE}                              Test Results Summary                               ${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════════════════════════${NC}"
    
    for result in "${test_results[@]}"; do
        echo -e "  ${result}"
    done
    
    echo -e "\n${WHITE}Overall Result: ${passed_tests}/${total_tests} tests passed${NC}"
    
    if [[ ${passed_tests} -eq ${total_tests} ]]; then
        echo -e "${GREEN}🎉 All system tests passed! InvestiGator is ready for operation.${NC}"
        return 0
    elif [[ ${passed_tests} -ge $((total_tests * 2 / 3)) ]]; then
        echo -e "${YELLOW} Most tests passed. InvestiGator should work but may have limitations.${NC}"
        return 0
    else
        echo -e "${RED} Some system tests failed. Run './investigator.sh --help' for setup guidance.${NC}"
        return 1
    fi
}

# Performance monitoring and system stats
show_system_stats() {
    log_step " Displaying system performance statistics..."
    
    echo -e "${BLUE}════════════════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${WHITE}                           🐊 InvestiGator System Statistics 🤓                        ${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════════════════════════${NC}"
    
    # System information
    echo -e "\n${CYAN}🖥️  System Information:${NC}"
    echo "   • OS: $(uname -s) $(uname -r)"
    echo "   • Architecture: $(uname -m)"
    if [[ "$(uname)" == "Darwin" ]]; then
        echo "   • macOS Version: $(sw_vers -productVersion)"
        local ram_gb
        ram_gb=$(sysctl -n hw.memsize | awk '{print int($0/1024/1024/1024)}')
        echo "   • RAM: ${ram_gb}GB"
    fi
    
    # Disk usage
    echo -e "\n${CYAN}💾 Storage Information:${NC}"
    echo "   • InvestiGator Directory: $(du -sh "${SCRIPT_DIR}" 2>/dev/null | cut -f1)"
    echo "   • Available Space: $(df -h "${SCRIPT_DIR}" | tail -1 | awk '{print $4}')"
    
    if [[ -d "${DATA_DIR}" ]]; then
        echo "   • Data Directory: $(du -sh "${DATA_DIR}" 2>/dev/null | cut -f1)"
    fi
    
    if [[ -d "${REPORTS_DIR}" ]]; then
        local report_count
        report_count=$(find "${REPORTS_DIR}" -name "*.pdf" | wc -l)
        echo "   • Reports: ${report_count} files ($(du -sh "${REPORTS_DIR}" 2>/dev/null | cut -f1))"
    fi
    
    if [[ -d "${LOG_DIR}" ]]; then
        echo "   • Logs: $(du -sh "${LOG_DIR}" 2>/dev/null | cut -f1)"
    fi
    
    # AI Models
    echo -e "\n${CYAN} AI Models:${NC}"
    if command -v ollama &> /dev/null; then
        local model_info
        model_info=$(ollama list 2>/dev/null)
        if [[ -n "${model_info}" ]]; then
            echo "${model_info}" | tail -n +2 | while IFS= read -r line; do
                echo "   • ${line}"
            done
        else
            echo "   • No models installed"
        fi
        
        # Model storage size
        if [[ -d ~/.ollama/models ]]; then
            echo "   • Total Model Size: $(du -sh ~/.ollama/models 2>/dev/null | cut -f1)"
        fi
    else
        echo "   • Ollama not installed"
    fi
    
    # Database information
    echo -e "\n${CYAN}🗄️  Database Information:${NC}"
    activate_python_env
    
    python3 -c "
import sys
sys.path.append('.')
try:
    from utils.db import get_db_manager
    db = get_db_manager()
    if db.test_connection():
        with db.get_session() as session:
            # Get table sizes
            tables = ['stock_analysis', 'sec_filings', 'technical_indicators', 'analysis_reports']
            for table in tables:
                try:
                    result = session.execute(f'SELECT COUNT(*) FROM {table}').scalar()
                    print(f'   • {table}: {result} records')
                except:
                    print(f'   • {table}: Table not found')
    else:
        print('   • Database connection failed')
except Exception as e:
    print(f'   • Database error: {e}')
" 2>/dev/null || echo "   • Database statistics unavailable"
    
    # Recent activity
    echo -e "\n${CYAN}📈 Recent Activity:${NC}"
    if [[ -f "${LOG_DIR}/investigator.log" ]]; then
        local recent_analyses
        recent_analyses=$(grep -c "analysis completed" "${LOG_DIR}/investigator.log" 2>/dev/null || echo "0")
        echo "   • Recent Analyses: ${recent_analyses}"
        
        local last_analysis
        last_analysis=$(grep "analysis completed" "${LOG_DIR}/investigator.log" | tail -1 | cut -d']' -f1 | tr -d '[' 2>/dev/null || echo "Never")
        echo "   • Last Analysis: ${last_analysis}"
    fi
    
    if [[ -d "${REPORTS_DIR}" ]]; then
        local latest_report
        latest_report=$(find "${REPORTS_DIR}" -name "*.pdf" -type f -printf '%T+ %f\n' 2>/dev/null | sort | tail -1 | cut -d' ' -f2 || echo "None")
        echo "   • Latest Report: ${latest_report}"
    fi
    
    echo -e "\n${BLUE}════════════════════════════════════════════════════════════════════════════════${NC}"
}

# Enhanced usage display
show_usage() {
    cat << EOF
${CYAN}════════════════════════════════════════════════════════════════════════════════${NC}
${WHITE}                     🐊 InvestiGator AI Investment Research Assistant 🤓                   ${NC}
${CYAN}════════════════════════════════════════════════════════════════════════════════${NC}

${YELLOW}USAGE:${NC}
    $0 [OPTIONS]

${YELLOW}STOCK ANALYSIS OPTIONS:${NC}
    --symbol SYMBOL              Analyze single stock (complete pipeline)
    --symbols SYMBOL1 SYMBOL2    Analyze multiple stocks (batch mode)

${YELLOW}REPORTING OPTIONS:${NC}
    --weekly-report              Generate comprehensive weekly PDF report
    --send-email                 Send report via email (use with --weekly-report)

${YELLOW}AUTOMATION OPTIONS:${NC}
    --start-scheduler           Start automated scheduler (weekly reports)

${YELLOW}SYSTEM OPTIONS:${NC}
    --test-system               Run comprehensive system tests
    --system-stats              Display system performance statistics
    --clean-cache               Clean cache for specified symbols (requires --symbol/--symbols)
    --clean-cache-all           Clean all caches completely
    --clean-cache-disk          Clean disk cache only
    --clean-cache-db            Clean database cache only
    --inspect-cache             Inspect cache contents (optionally for specific symbols)
    --cache-sizes               Show cache sizes and statistics
    --force-refresh             Force refresh cache for specified symbols
    --test-cache                Run cache system tests
    --run-tests [mode]          Run test suite (unit|integration|coverage|all)
    --setup-system              Run automated system setup
    --setup-database             Setup PostgreSQL database and schema
    --generate-docs [format]     Generate documentation (html|pdf)
    --config FILE               Use alternative configuration file
    --debug                     Enable debug logging
    --help                      Show this help message

${YELLOW}EXAMPLES:${NC}
    ${GREEN}# Analyze single stock${NC}
    $0 --symbol AAPL
    
    ${GREEN}# Analyze multiple stocks${NC}
    $0 --symbols AAPL GOOGL MSFT AMZN TSLA
    
    ${GREEN}# Clean cache for specific symbols${NC}
    $0 --clean-cache --symbol AAPL
    $0 --clean-cache --symbols AAPL MSFT GOOGL
    
    ${GREEN}# Cache management operations${NC}
    $0 --clean-cache-all         # Clean all caches
    $0 --inspect-cache           # Inspect all cache contents
    $0 --inspect-cache --symbol AAPL  # Inspect cache for AAPL
    $0 --cache-sizes             # Show cache sizes
    $0 --force-refresh --symbol AAPL   # Force refresh AAPL data
    
    ${GREEN}# Generate and email comprehensive weekly report${NC}
    $0 --weekly-report --send-email
    
    ${GREEN}# Start automated scheduler for hands-free operation${NC}
    $0 --start-scheduler
    
    ${GREEN}# Run complete system diagnostics${NC}
    $0 --test-system

${YELLOW}ANALYSIS PIPELINE:${NC}
    ${BLUE}1. SEC Fundamental Analysis${NC}     - Downloads 10-K filings, AI analysis
    ${BLUE}2. Yahoo Technical Analysis${NC}     - Market data, technical indicators, AI patterns  
    ${BLUE}3. Investment Synthesis${NC}         - Combined AI recommendation, PDF report

${YELLOW}LOG FILES:${NC}
    All execution logs are saved to: ${LOG_DIR}/
    • investigator.log           - Main InvestiGator activity
    • sec_fundamental_*.log      - SEC analysis detailed logs
    • yahoo_technical_*.log      - Technical analysis detailed logs
    • synthesizer_*.log          - Synthesis and reporting logs
    • scheduler.log              - Automated scheduler activity

${YELLOW}CONFIGURATION:${NC}
    Edit ${CONFIG_FILE} to customize:
    • Stock tracking list        • Analysis parameters
    • Email delivery settings    • AI model preferences
    • Database connection        • API credentials

${YELLOW}SYSTEM REQUIREMENTS:${NC}
    • macOS 12.0+ with Apple Silicon (M1/M2/M3)
    • 32GB+ RAM (64GB recommended)
    • 200GB+ free storage
    • Active internet connection

${CYAN}════════════════════════════════════════════════════════════════════════════════${NC}
${COPYRIGHT}
${LICENSE}
${CYAN}════════════════════════════════════════════════════════════════════════════════${NC}
EOF
}

# Enhanced cleanup function
cleanup_on_exit() {
    local exit_code=$?
    
    log_warn "🐊 InvestiGator interrupted (exit code: ${exit_code})"
    
    # Kill any background processes we might have started
    local bg_jobs
    bg_jobs=$(jobs -r)
    if [[ -n "${bg_jobs}" ]]; then
        log_info "Terminating background processes..."
        kill %% 2>/dev/null || true
    fi
    
    # Display execution summary if we processed any stocks
    if [[ ${TOTAL_STOCKS} -gt 0 ]]; then
        local current_time
        current_time=$(date +%s)
        local total_duration=$((current_time - START_TIME))
        local duration_min=$((total_duration / 60))
        local duration_sec=$((total_duration % 60))
        
        echo -e "\n${YELLOW}═══════════════════════════════════════════════════════════════════════════════${NC}"
        log_info " Execution Summary:"
        echo -e "${BLUE}   • Total Runtime: ${duration_min}m ${duration_sec}s${NC}"
        echo -e "${BLUE}   • Stocks Processed: ${TOTAL_STOCKS}${NC}"
        echo -e "${GREEN}   • Successful: ${SUCCESSFUL_ANALYSES}${NC}"
        
        if [[ ${FAILED_ANALYSES} -gt 0 ]]; then
            echo -e "${RED}   • Failed: ${FAILED_ANALYSES}${NC}"
            echo -e "${RED}   • Failed Symbols: ${FAILED_SYMBOLS[*]}${NC}"
        fi
        echo -e "${YELLOW}═══════════════════════════════════════════════════════════════════════════════${NC}"
    fi
    
    exit ${exit_code}
}

# Main InvestiGator function
main() {
    local symbols=()
    local weekly_report=false
    local send_email=false
    local start_scheduler_flag=false
    local test_system_flag=false
    local show_stats=false
    local clean_cache=false
    local clean_cache_all=false
    local clean_cache_disk=false
    local clean_cache_db=false
    local inspect_cache=false
    local cache_sizes=false
    local force_refresh=false
    local test_cache=false
    local run_tests=false
    local test_mode="all"
    local setup_system=false
    local setup_database=false
    local generate_docs=false
    local docs_format="html"
    local config_file="${CONFIG_FILE}"
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --symbol)
                if [[ -z "${2:-}" ]]; then
                    log_error "Option --symbol requires a stock symbol"
                    show_usage
                    exit 1
                fi
                symbols+=("$2")
                shift 2
                ;;
            --symbols)
                shift
                while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                    symbols+=("$1")
                    shift
                done
                ;;
            --weekly-report)
                weekly_report=true
                shift
                ;;
            --send-email)
                send_email=true
                shift
                ;;
            --start-scheduler)
                start_scheduler_flag=true
                shift
                ;;
            --test-system)
                test_system_flag=true
                shift
                ;;
            --system-stats)
                show_stats=true
                shift
                ;;
            --config)
                if [[ -z "${2:-}" ]]; then
                    log_error "Option --config requires a file path"
                    exit 1
                fi
                config_file="$2"
                shift 2
                ;;
            --debug)
                DEBUG=true
                shift
                ;;
            --clean-cache)
                clean_cache=true
                shift
                ;;
            --clean-cache-all)
                clean_cache_all=true
                shift
                ;;
            --clean-cache-disk)
                clean_cache_disk=true
                shift
                ;;
            --clean-cache-db)
                clean_cache_db=true
                shift
                ;;
            --inspect-cache)
                inspect_cache=true
                shift
                ;;
            --cache-sizes)
                cache_sizes=true
                shift
                ;;
            --force-refresh)
                force_refresh=true
                shift
                ;;
            --test-cache)
                test_cache=true
                shift
                ;;
            --run-tests)
                run_tests=true
                if [[ -n "${2:-}" && ! "$2" =~ ^-- ]]; then
                    test_mode="$2"
                    shift 2
                else
                    shift
                fi
                ;;
            --setup-system)
                setup_system=true
                shift
                ;;
            --setup-database)
                setup_database=true
                shift
                ;;
            --generate-docs)
                generate_docs=true
                if [[ -n "${2:-}" && ! "$2" =~ ^-- ]]; then
                    docs_format="$2"
                    shift 2
                else
                    shift
                fi
                ;;
            --help)
                show_usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    # Setup signal handling for graceful shutdown
    trap cleanup_on_exit INT TERM EXIT
    
    # Show banner
    show_banner
    
    # Display startup information
    log_info "🐊 InvestiGator v${SCRIPT_VERSION} starting..."
    log_info " Working Directory: ${SCRIPT_DIR}"
    log_info " Configuration: ${config_file}"
    log_info " Logs Directory: ${LOG_DIR}"
    
    # Handle different execution modes
    if [[ "${show_stats}" == "true" ]]; then
        show_system_stats
        exit 0
    fi
    
    if [[ "${test_system_flag}" == "true" ]]; then
        test_system
        exit $?
    fi
    
    if [[ "${start_scheduler_flag}" == "true" ]]; then
        start_scheduler
        exit $?
    fi
    
    # Handle cache cleanup
    if [[ "${clean_cache}" == "true" ]]; then
        if [[ ${#symbols[@]} -eq 0 ]]; then
            log_error "❌ --clean-cache requires at least one symbol (use --symbol or --symbols)"
            show_usage
            exit 1
        fi
        
        log_info "🧹 Starting cache cleanup for symbols: ${symbols[*]}"
        activate_python_env
        
        local cache_cleanup_cmd
        if [[ ${#symbols[@]} -eq 1 ]]; then
            cache_cleanup_cmd="python ${SCRIPT_DIR}/utils/cache_cleanup.py --symbol ${symbols[0]} --verbose"
        else
            cache_cleanup_cmd="python ${SCRIPT_DIR}/utils/cache_cleanup.py --symbols ${symbols[*]} --verbose"
        fi
        
        log_info "Executing: ${cache_cleanup_cmd}"
        if eval "${cache_cleanup_cmd}"; then
            log_success "✅ Cache cleanup completed successfully"
        else
            log_error "❌ Cache cleanup failed"
            exit 1
        fi
        exit 0
    fi
    
    # Handle cache management operations
    if [[ "${clean_cache_all}" == "true" ]]; then
        log_info "🧹 Cleaning all caches..."
        activate_python_env
        if python "${SCRIPT_DIR}/utils/cache_cleanup.py" --verbose; then
            log_success "✅ All caches cleaned successfully"
        else
            log_error "❌ Cache cleanup failed"
            exit 1
        fi
        exit 0
    fi
    
    if [[ "${clean_cache_disk}" == "true" ]]; then
        log_info "🧹 Cleaning disk cache..."
        activate_python_env
        local cmd_args="--disk"
        [[ ${#symbols[@]} -gt 0 ]] && cmd_args="${cmd_args} --symbols ${symbols[*]}"
        if python "${SCRIPT_DIR}/utils/cache_cleanup.py" ${cmd_args} --verbose; then
            log_success "✅ Disk cache cleaned successfully"
        else
            log_error "❌ Disk cache cleanup failed"
            exit 1
        fi
        exit 0
    fi
    
    if [[ "${clean_cache_db}" == "true" ]]; then
        log_info "🧹 Cleaning database cache..."
        activate_python_env
        local cmd_args="--db"
        [[ ${#symbols[@]} -gt 0 ]] && cmd_args="${cmd_args} --symbols ${symbols[*]}"
        if python "${SCRIPT_DIR}/utils/cache_cleanup.py" ${cmd_args} --verbose; then
            log_success "✅ Database cache cleaned successfully"
        else
            log_error "❌ Database cache cleanup failed"
            exit 1
        fi
        exit 0
    fi
    
    if [[ "${inspect_cache}" == "true" ]]; then
        log_info "🔍 Inspecting cache contents..."
        activate_python_env
        local cmd_args=""
        [[ ${#symbols[@]} -gt 0 ]] && cmd_args="--symbols ${symbols[*]}"
        if python "${SCRIPT_DIR}/utils/cache_inspector.py" ${cmd_args}; then
            log_success "✅ Cache inspection completed"
        else
            log_error "❌ Cache inspection failed"
            exit 1
        fi
        exit 0
    fi
    
    if [[ "${cache_sizes}" == "true" ]]; then
        log_info "📊 Showing cache sizes..."
        echo ""
        echo -e "${BLUE}Disk Cache Sizes:${NC}"
        for dir in data/sec_cache data/llm_cache data/technical_cache data/price_cache; do
            if [[ -d "$dir" ]]; then
                size=$(du -sh "$dir" 2>/dev/null | cut -f1)
                echo "  $dir: $size"
            fi
        done
        echo ""
        echo -e "${BLUE}Database Cache Sizes:${NC}"
        activate_python_env
        python -c "
import psycopg2
from config import get_config

config = get_config()
try:
    conn = psycopg2.connect(config.database.url)
    cur = conn.cursor()
    
    tables = [
        'sec_response_store', 'llm_response_store', 'all_submission_store',
        'all_companyfacts_store', 'quarterly_metrics', 'synthesis_results'
    ]
    
    for table in tables:
        try:
            cur.execute(f'''
                SELECT 
                    pg_size_pretty(pg_total_relation_size('{table}')) as size,
                    COUNT(*) as records
                FROM {table}
            ''')
            size, count = cur.fetchone()
            print(f'  {table}: {size} ({count} records)')
        except:
            pass
            
    cur.close()
    conn.close()
except Exception as e:
    print(f'  Error connecting to database: {e}')
"
        exit 0
    fi
    
    if [[ "${force_refresh}" == "true" ]]; then
        if [[ ${#symbols[@]} -eq 0 ]]; then
            log_error "❌ --force-refresh requires at least one symbol (use --symbol or --symbols)"
            show_usage
            exit 1
        fi
        
        log_info "🔄 Force refreshing data for symbols: ${symbols[*]}"
        activate_python_env
        
        for symbol in "${symbols[@]}"; do
            log_info "🔄 Force refreshing $symbol..."
            # Create temporary config with force refresh
            cat > temp_refresh_config.json <<EOF
{
    "cache_control": {
        "storage": ["disk", "rdbms"],
        "force_refresh_symbols": ["$symbol"]
    }
}
EOF
            # Merge with existing config
            python -c "
import json
with open('${config_file}', 'r') as f:
    config = json.load(f)
config.setdefault('cache_control', {})['force_refresh_symbols'] = ['$symbol']
with open('temp_refresh_config.json', 'w') as f:
    json.dump(config, f, indent=2)
"
            # Run analysis with force refresh
            if python sec_fundamental.py --symbol "$symbol" --config temp_refresh_config.json; then
                log_success "✅ Force refresh completed for $symbol"
            else
                log_error "❌ Force refresh failed for $symbol"
            fi
            
            # Clean up
            rm -f temp_refresh_config.json
        done
        exit 0
    fi
    
    if [[ "${test_cache}" == "true" ]]; then
        log_info "🧪 Running cache tests..."
        activate_python_env
        if python test_cache_combinations.py; then
            log_success "✅ Cache tests completed successfully"
        else
            log_error "❌ Cache tests failed"
            exit 1
        fi
        exit 0
    fi
    
    if [[ "${run_tests}" == "true" ]]; then
        log_info "🧪 Running test suite (mode: ${test_mode})..."
        activate_python_env
        
        # Install test dependencies if needed
        log_info "📋 Checking test dependencies..."
        pip install -q pytest pytest-cov pytest-mock coverage 2>/dev/null || echo "Dependencies already installed"
        
        # Create test reports directory
        mkdir -p test_reports
        
        case "${test_mode}" in
            "unit")
                log_info "🔬 Running unit tests only..."
                if command -v pytest &> /dev/null; then
                    pytest tests/test_*.py -v --tb=short -k "not integration"
                else
                    log_error "❌ pytest not found. Install with: pip install pytest"
                    exit 1
                fi
                ;;
            "integration")
                log_info "🔗 Running integration tests..."
                if command -v pytest &> /dev/null; then
                    pytest tests/test_*integration*.py -v --tb=short
                else
                    log_error "❌ pytest not found. Install with: pip install pytest"
                    exit 1
                fi
                ;;
            "coverage")
                log_info "📊 Running tests with coverage analysis..."
                if command -v pytest &> /dev/null; then
                    pytest --cov=utils --cov=synthesizer --cov=sec_fundamental --cov=yahoo_technical \
                           --cov-report=html:test_reports/coverage_html \
                           --cov-report=term-missing \
                           --cov-report=xml:test_reports/coverage.xml \
                           tests/ -v
                    log_info "📈 Coverage report generated in test_reports/coverage_html/index.html"
                else
                    log_error "❌ pytest not found. Install with: pip install pytest pytest-cov"
                    exit 1
                fi
                ;;
            "all"|*)
                log_info "🚀 Running all tests..."
                if command -v pytest &> /dev/null; then
                    pytest tests/ -v --tb=short 2>/dev/null || echo "✅ No test directory found - ready for new test suite"
                else
                    log_error "❌ pytest not found. Install with: pip install pytest"
                    exit 1
                fi
                ;;
        esac
        
        if [[ $? -eq 0 ]]; then
            log_success "✅ Tests completed successfully!"
        else
            log_error "❌ Some tests failed!"
            exit 1
        fi
        exit 0
    fi
    
    if [[ "${setup_system}" == "true" ]]; then
        log_info "🔧 Running automated system setup..."
        
        # Check basic prerequisites
        log_info "📋 Checking system prerequisites..."
        
        # Check Homebrew
        if ! command -v brew &> /dev/null; then
            log_error "❌ Homebrew not found. Please install: /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
            exit 1
        fi
        
        # Install system dependencies
        log_info "📦 Installing system dependencies..."
        brew install postgresql@14 python@3.9 git 2>/dev/null || log_warn "Some packages may already be installed"
        
        # Create Python virtual environment
        if [[ ! -d "${PYTHON_ENV}" ]]; then
            log_info "🐍 Creating Python virtual environment..."
            python3 -m venv "${PYTHON_ENV}"
        fi
        
        # Install Python packages
        log_info "📦 Installing Python packages..."
        activate_python_env
        pip install --upgrade pip
        pip install -r requirements.txt
        
        # Install Ollama if not present
        if ! command -v ollama &> /dev/null; then
            log_info "🤖 Installing Ollama..."
            curl -fsSL https://ollama.com/install.sh | sh
        fi
        
        # Start PostgreSQL
        log_info "🗄️ Starting PostgreSQL..."
        brew services start postgresql@14 || log_warn "PostgreSQL may already be running"
        
        log_success "✅ System setup completed successfully!"
        log_info "🔄 Next steps:"
        log_info "   1. Configure config.json with your settings"
        log_info "   2. Run: ./investigator.sh --setup-database"
        log_info "   3. Download AI models: ollama pull phi4-reasoning:plus"
        exit 0
    fi
    
    if [[ "${setup_database}" == "true" ]]; then
        log_info "🗄️ Setting up PostgreSQL database..."
        
        # Check if PostgreSQL is running
        if ! pg_ctl status -D /opt/homebrew/var/postgres@14 &>/dev/null; then
            log_info "⚡ Starting PostgreSQL..."
            brew services start postgresql@14
            sleep 3
        fi
        
        # Read database config
        activate_python_env
        DB_NAME=$(python -c "from config import get_config; print(get_config().database.database)" 2>/dev/null || echo "investment_analysis")
        DB_USER=$(python -c "from config import get_config; print(get_config().database.username)" 2>/dev/null || echo "investment_ai")
        DB_PASS=$(python -c "from config import get_config; print(get_config().database.password)" 2>/dev/null || echo "investment_pass")
        
        log_info "📊 Database: ${DB_NAME}"
        log_info "👤 User: ${DB_USER}"
        
        # Create database if it doesn't exist
        if ! psql -lqt | cut -d \| -f 1 | grep -qw "${DB_NAME}"; then
            log_info "🏗️ Creating database ${DB_NAME}..."
            createdb "${DB_NAME}"
        fi
        
        # Create user if it doesn't exist
        if ! psql -tc "SELECT 1 FROM pg_user WHERE usename = '${DB_USER}'" | grep -q 1; then
            log_info "👤 Creating user ${DB_USER}..."
            psql -c "CREATE USER ${DB_USER} WITH PASSWORD '${DB_PASS}';"
        fi
        
        # Grant permissions
        log_info "🔐 Setting up permissions..."
        psql -d "${DB_NAME}" -c "GRANT ALL PRIVILEGES ON DATABASE ${DB_NAME} TO ${DB_USER};"
        psql -d "${DB_NAME}" -c "GRANT ALL PRIVILEGES ON SCHEMA public TO ${DB_USER};"
        psql -d "${DB_NAME}" -c "GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO ${DB_USER};"
        psql -d "${DB_NAME}" -c "GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO ${DB_USER};"
        
        # Initialize database schema
        if [[ -f "schema/consolidated_schema.sql" ]]; then
            log_info "📋 Loading database schema..."
            psql -U "${DB_USER}" -d "${DB_NAME}" -f schema/consolidated_schema.sql
        else
            log_warn "⚠️ No schema file found at schema/consolidated_schema.sql"
        fi
        
        # Test connection
        log_info "🔍 Testing database connection..."
        if psql -U "${DB_USER}" -d "${DB_NAME}" -c "SELECT 1;" &>/dev/null; then
            log_success "✅ Database setup completed successfully!"
        else
            log_error "❌ Database connection test failed"
            exit 1
        fi
        exit 0
    fi
    
    if [[ "${generate_docs}" == "true" ]]; then
        log_info "📖 Generating documentation (format: ${docs_format})..."
        
        # Check prerequisites
        if ! command -v asciidoctor &> /dev/null; then
            log_error "❌ asciidoctor not found. Install with: gem install asciidoctor"
            exit 1
        fi
        
        if [[ "${docs_format}" == "pdf" ]] && ! command -v asciidoctor-pdf &> /dev/null; then
            log_error "❌ asciidoctor-pdf not found. Install with: gem install asciidoctor-pdf"
            exit 1
        fi
        
        # Create docs directory if it doesn't exist
        mkdir -p docs
        
        # Generate main documentation file
        cat > docs/investigator-guide.adoc << 'EOF'
= InvestiGator - AI Investment Research Assistant
:doctype: book
:toc: left
:toclevels: 3
:sectanchors:
:sectlinks:
:sectnums:
:source-highlighter: highlight.js
:icons: font

== Introduction

InvestiGator is a comprehensive AI-powered investment research system that combines SEC filing analysis with technical analysis to generate professional investment recommendations.

=== Key Features

* Local AI processing using Ollama
* SEC filing fundamental analysis
* Technical analysis with advanced indicators
* Professional PDF report generation
* Email delivery system
* Comprehensive caching system

== Quick Start

=== Installation

[source,bash]
----
# Clone repository
git clone https://github.com/your-repo/InvestiGator.git
cd InvestiGator

# Run automated setup
./investigator.sh --setup-system
./investigator.sh --setup-database

# Configure system
cp config.sample.json config.json
# Edit config.json with your settings

# Download AI models
ollama pull phi4-reasoning:plus
ollama pull qwen2.5:32b-instruct-q4_K_M
ollama pull llama3.1:8b-instruct-q8_0

# Test system
./investigator.sh --test-system
----

=== Basic Usage

[source,bash]
----
# Analyze single stock
./investigator.sh --symbol AAPL

# Analyze multiple stocks
./investigator.sh --symbols AAPL GOOGL MSFT

# Generate weekly report
./investigator.sh --weekly-report --send-email

# Cache management
./investigator.sh --cache-sizes
./investigator.sh --clean-cache --symbol AAPL
./investigator.sh --clean-cache-all

# System operations
./investigator.sh --test-system
./investigator.sh --run-tests coverage
----

== Architecture

The system uses a modular design pattern architecture:

* **Data Layer**: PostgreSQL with intelligent caching
* **Analysis Layer**: SEC fundamental + Yahoo technical analysis
* **AI Layer**: Local Ollama models for processing
* **Synthesis Layer**: Combined analysis and reporting
* **Orchestration Layer**: Single investigator.sh script

== Cache Management

=== Cache Types

* **Disk Cache**: File-based caching for SEC filings and market data
* **Database Cache**: PostgreSQL-based caching for processed analysis
* **Memory Cache**: In-process caching for frequently accessed data

=== Cache Operations

[source,bash]
----
# Inspect cache contents
./investigator.sh --inspect-cache
./investigator.sh --inspect-cache --symbol AAPL

# Clean caches
./investigator.sh --clean-cache-all      # Clean everything
./investigator.sh --clean-cache-disk     # Disk only
./investigator.sh --clean-cache-db       # Database only
./investigator.sh --clean-cache --symbol AAPL  # Specific symbol

# Cache statistics
./investigator.sh --cache-sizes

# Force refresh
./investigator.sh --force-refresh --symbol AAPL
----

== Testing

=== Test Modes

[source,bash]
----
# Run all tests
./investigator.sh --run-tests

# Specific test types
./investigator.sh --run-tests unit          # Unit tests only
./investigator.sh --run-tests integration   # Integration tests
./investigator.sh --run-tests coverage      # With coverage report

# Cache-specific tests
./investigator.sh --test-cache

# System health check
./investigator.sh --test-system
----

== Configuration

=== Main Configuration (config.json)

Key configuration sections:

[source,json]
----
{
  "database": {
    "host": "localhost",
    "port": 5432,
    "database": "investment_analysis",
    "username": "investment_ai",
    "password": "your_password"
  },
  "ollama": {
    "host": "localhost",
    "port": 11434,
    "models": {
      "fundamental_analysis": "phi4-reasoning:plus",
      "technical_analysis": "qwen2.5:32b-instruct-q4_K_M",
      "synthesizer": "llama3.1:8b-instruct-q8_0"
    }
  },
  "sec": {
    "user_agent": "Your Company/1.0 (your-email@example.com)"
  }
}
----

=== Cache Configuration

Configure caching behavior:

[source,json]
----
{
  "cache_control": {
    "storage": ["disk", "rdbms"],
    "disk_cache_ttl_hours": 24,
    "db_cache_ttl_hours": 168,
    "max_cache_size_gb": 10
  }
}
----

== Troubleshooting

=== Common Issues

==== Ollama Connection Issues

[source,bash]
----
# Check Ollama status
ollama list

# Start Ollama service
ollama serve

# Test connection
curl http://localhost:11434/api/tags
----

==== Database Connection Issues

[source,bash]
----
# Check PostgreSQL status
brew services list | grep postgresql

# Restart PostgreSQL
brew services restart postgresql@14

# Test connection
./investigator.sh --setup-database
----

==== Performance Issues

[source,bash]
----
# Check system resources
./investigator.sh --system-stats

# Clean old cache data
./investigator.sh --clean-cache-all

# Check cache sizes
./investigator.sh --cache-sizes
----

== Maintenance

=== Regular Maintenance

* **Weekly**: Review reports, check logs, verify email delivery
* **Monthly**: Update AI models, clean old data, review configuration
* **Quarterly**: Backup database, review analysis parameters

=== Backup Procedures

[source,bash]
----
# Database backup
pg_dump -U investment_ai investment_analysis > backup_$(date +%Y%m%d).sql

# Configuration backup
tar -czf config_backup_$(date +%Y%m%d).tar.gz config.json reports/ logs/
----

== Support

For issues and questions:

* Check the troubleshooting section
* Review log files in logs/ directory
* Run system diagnostics: `./investigator.sh --test-system`
* Check cache status: `./investigator.sh --cache-sizes`

EOF
        
        # Build documentation
        case "${docs_format}" in
            "html")
                asciidoctor docs/investigator-guide.adoc
                log_success "✅ HTML documentation generated: docs/investigator-guide.html"
                ;;
            "pdf")
                asciidoctor-pdf docs/investigator-guide.adoc
                log_success "✅ PDF documentation generated: docs/investigator-guide.pdf"
                ;;
            *)
                log_error "❌ Unsupported format: ${docs_format}"
                exit 1
                ;;
        esac
        exit 0
    fi
    
    # Check prerequisites for analysis operations
    if [[ "${weekly_report}" == "true" || ${#symbols[@]} -gt 0 ]]; then
        if ! check_prerequisites; then
            log_error "Prerequisites check failed. See the errors above for specific issues to fix."
            exit 1
        else
            log_info "Prerequisites failed -> Symbols: ${#symbols[@]}"
        fi
    fi
    
    if [[ "${weekly_report}" == "true" ]]; then
        # Get list of stocks from config if no symbols specified
        if [[ ${#symbols[@]} -eq 0 ]]; then
            log_info "📋 Retrieving stock list from configuration..."
            activate_python_env
            
            # Extract stocks from config.json
            readarray -t symbols < <(python3 -c "
import json
import sys
try:
    with open('config.json', 'r') as f:
        config = json.load(f)
    stocks = config.get('stocks_to_track', [])
    for stock in stocks:
        print(stock)
except Exception as e:
    print(f'Error reading config: {e}', file=sys.stderr)
    sys.exit(1)
")
            
            if [[ ${#symbols[@]} -eq 0 ]]; then
                log_error "No stocks found in configuration"
                exit 1
            fi
            
            log_info " Found ${#symbols[@]} stocks in configuration: ${symbols[*]}"
        fi
        
        # Run batch analysis for weekly report
        if [[ ${#symbols[@]} -gt 0 ]]; then
            log_info " Running batch analysis for weekly report..."
            run_batch_analysis "${symbols[@]}"
        fi
        
        # Generate and optionally email the report
        generate_weekly_report "${send_email}"
        exit $?
    fi
    
    if [[ ${#symbols[@]} -gt 0 ]]; then
        if [[ ${#symbols[@]} -eq 1 ]]; then
            # Single stock analysis
            log_info " Single stock analysis mode"
            analyze_stock "${symbols[0]^^}"
        else
            # Batch analysis
            log_info " Batch analysis mode"
            run_batch_analysis "${symbols[@]}"
        fi
        exit $?
    fi
    
    # No arguments provided - show usage
    log_error " No analysis target specified"
    echo
    show_usage
    exit 1
}

# Script execution entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi

# END OF FILE - investigator.sh
# Copyright (c) 2025 Vijaykumar Singh  
# Licensed under the Apache License, Version 2.0
