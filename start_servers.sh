NUM_SERVERS=${1:-4}
ACTION=${2:-start}
BASE_PORT=8000
SHOWDOWN_DIR="."  # Scripts are inside the showdown directory

# PID file to track servers
PID_FILE="/tmp/pokemon_showdown_pids.txt"

stop_servers() {
    if [ -f "$PID_FILE" ]; then
        echo "Stopping Pokemon Showdown servers..."
        while read pid; do
            if kill -0 "$pid" 2>/dev/null; then
                kill "$pid"
                echo "  Stopped PID $pid"
            fi
        done < "$PID_FILE"
        rm -f "$PID_FILE"
        echo "All servers stopped."
    else
        echo "No PID file found. Killing any node pokemon-showdown processes..."
        pkill -f "pokemon-showdown start" 2>/dev/null
        echo "Done."
    fi
}

start_servers() {
    # Stop any existing servers first
    stop_servers 2>/dev/null
    
    > "$PID_FILE"
    
    echo "Starting $NUM_SERVERS Pokemon Showdown servers..."
    echo ""
    
    for i in $(seq 0 $((NUM_SERVERS - 1))); do
        PORT=$((BASE_PORT + i))
        
        # Start server in background
        cd "$SHOWDOWN_DIR" 2>/dev/null || {
            echo "Error: Directory '$SHOWDOWN_DIR' not found."
            echo "Edit SHOWDOWN_DIR in this script to point to your pokemon-showdown directory."
            exit 1
        }
        
        node pokemon-showdown start --no-security --port "$PORT" > "/tmp/showdown_${PORT}.log" 2>&1 &
        PID=$!
        echo "$PID" >> "$PID_FILE"
        
        cd - > /dev/null
        
        echo "  Server $((i + 1)): port $PORT (PID $PID)"
    done
    
    # Wait for servers to be ready
    echo ""
    echo "Waiting for servers to start..."
    sleep 3
    
    # Verify each server
    ALL_OK=true
    for i in $(seq 0 $((NUM_SERVERS - 1))); do
        PORT=$((BASE_PORT + i))
        if curl -s "http://localhost:$PORT" > /dev/null 2>&1; then
            echo "  ✓ Port $PORT ready"
        else
            echo "  ✗ Port $PORT not responding (check /tmp/showdown_${PORT}.log)"
            ALL_OK=false
        fi
    done
    
    echo ""
    if $ALL_OK; then
        echo "All $NUM_SERVERS servers running!"
    else
        echo "Some servers failed to start. Check logs in /tmp/showdown_*.log"
    fi
    
    echo ""
    echo "To stop all servers:  $0 $NUM_SERVERS stop"
    echo "Server logs:          /tmp/showdown_800X.log"
}

case "$ACTION" in
    start)
        start_servers
        ;;
    stop)
        stop_servers
        ;;
    *)
        echo "Usage: $0 [num_servers] [start|stop]"
        exit 1
        ;;
esac