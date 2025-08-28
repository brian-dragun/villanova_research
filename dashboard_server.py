#!/usr/bin/env python3
"""
Dashboard Web Server

This script starts a web server to make the research dashboard accessible over the network.
It allows you to access your dashboard remotely using your server's IP address.

Usage:
  python dashboard_server.py [--port 8080] [--host 0.0.0.0]
"""

import os
import sys
import argparse
import http.server
import socketserver
from urllib.parse import urlparse, parse_qs
import json
import time
from pathlib import Path
from output_manager import OutputManager

class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    """Custom request handler for the dashboard server"""
    
    def __init__(self, *args, **kwargs):
        self.output_mgr = OutputManager()
        self.dashboard_dir = self.output_mgr.dashboard_dir
        self.base_output_dir = self.output_mgr.base_output_dir
        super().__init__(*args, directory=self.base_output_dir, **kwargs)
    
    def do_GET(self):
        """Handle GET requests"""
        # Parse URL
        parsed_url = urlparse(self.path)
        path = parsed_url.path
        
        # Handle API requests
        if path.startswith('/api/'):
            self.handle_api_request(path[5:], parse_qs(parsed_url.query))
            return
            
        # Handle root path - redirect to dashboard
        if path == '/':
            self.send_response(302)  # Found/Redirect
            self.send_header('Location', '/_dashboard/dashboard.html')
            self.end_headers()
            return
            
        # Ensure dashboard is updated
        if path == '/_dashboard/dashboard.html':
            # Regenerate dashboard if it doesn't exist or is older than 5 minutes
            dashboard_path = os.path.join(self.dashboard_dir, 'dashboard.html')
            if not os.path.exists(dashboard_path) or time.time() - os.path.getmtime(dashboard_path) > 300:
                self.output_mgr._update_dashboard_index()
        
        # Default behavior - let SimpleHTTPRequestHandler handle it
        return super().do_GET()
    
    def handle_api_request(self, api_path, query_params):
        """Handle API requests"""
        if api_path == 'list':
            # Get filter params
            days = int(query_params.get('days', [None])[0]) if query_params.get('days') else None
            model = query_params.get('model', [None])[0]
            analysis_type = query_params.get('analysis', [None])[0]
            
            # Get outputs
            outputs = self.output_mgr.list_outputs(days=days, model=model, analysis_type=analysis_type)
            
            # Send response
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(outputs).encode())
            
        elif api_path == 'refresh':
            # Regenerate dashboard
            self.output_mgr._update_dashboard_index()
            
            # Send response
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok", "message": "Dashboard refreshed"}).encode())
            
        else:
            # Unknown API endpoint
            self.send_response(404)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": "API endpoint not found"}).encode())

def main():
    """Main function to start the dashboard server"""
    parser = argparse.ArgumentParser(
        description="Web server for Villanova Research Dashboard"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to run the server on (default: 8080)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address to bind to (default: 0.0.0.0 - all interfaces)"
    )
    
    args = parser.parse_args()
    
    # Ensure dashboard exists
    output_mgr = OutputManager()
    output_mgr._update_dashboard_index()
    
    # Create the server
    handler = DashboardHandler
    server = socketserver.TCPServer((args.host, args.port), handler)
    
    # Print info
    print(f"Starting dashboard server on http://{args.host}:{args.port}/")
    print(f"Dashboard URL: http://YOUR_SERVER_IP:{args.port}/")
    print("Press Ctrl+C to stop the server")
    
    try:
        # Run the server until interrupted
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.shutdown()
        server.server_close()

if __name__ == "__main__":
    main()