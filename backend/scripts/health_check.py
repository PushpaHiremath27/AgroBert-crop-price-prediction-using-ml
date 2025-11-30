"""
Health check and monitoring script.
Can be used to verify API availability and performance.
"""

import requests
import json
import time
import logging
from datetime import datetime
from typing import Dict, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HealthMonitor:
    """Monitor application health and performance."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api/v1"
        self.results = []
    
    def check_health(self) -> bool:
        """Check if API is healthy."""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            is_healthy = response.status_code == 200
            
            if is_healthy:
                logger.info("✓ API is healthy")
            else:
                logger.error(f"✗ API returned status {response.status_code}")
            
            return is_healthy
        
        except requests.exceptions.Timeout:
            logger.error("✗ API request timed out")
            return False
        except requests.exceptions.ConnectionError:
            logger.error("✗ Cannot connect to API")
            return False
        except Exception as e:
            logger.error(f"✗ Error: {e}")
            return False
    
    def check_endpoint(self, method: str, endpoint: str, params: Dict = None) -> bool:
        """Check a specific endpoint."""
        try:
            url = f"{self.api_url}{endpoint}"
            
            if method.upper() == "GET":
                response = requests.get(url, params=params, timeout=10)
            elif method.upper() == "POST":
                response = requests.post(url, params=params, json={}, timeout=10)
            else:
                return False
            
            is_ok = 200 <= response.status_code < 300
            
            status = "✓" if is_ok else "✗"
            logger.info(f"{status} {method} {endpoint}: {response.status_code}")
            
            return is_ok
        
        except Exception as e:
            logger.error(f"✗ {method} {endpoint}: {str(e)}")
            return False
    
    def test_endpoints(self):
        """Test key endpoints."""
        endpoints = [
            ("GET", "/health"),
            ("GET", "/status"),
            ("GET", "/agmarknet/prices", {"commodity": "wheat"}),
            ("GET", "/weather/current", {"location": "Delhi"}),
            ("GET", "/news/latest"),
            ("POST", "/predict"),
            ("GET", "/xai/shap-summary", {"commodity": "wheat"}),
        ]
        
        logger.info("\nTesting endpoints...")
        results = {
            "total": len(endpoints),
            "success": 0,
            "failed": 0,
            "endpoints": []
        }
        
        for method, endpoint, *params in endpoints:
            params = params[0] if params else None
            is_ok = self.check_endpoint(method, endpoint, params)
            
            results["success"] += 1 if is_ok else 0
            results["failed"] += 0 if is_ok else 1
            
            results["endpoints"].append({
                "method": method,
                "endpoint": endpoint,
                "status": "OK" if is_ok else "FAILED"
            })
        
        return results
    
    def performance_test(self, endpoint: str = "/health", iterations: int = 10) -> Dict:
        """Test endpoint performance."""
        logger.info(f"\nPerformance test ({iterations} iterations)...")
        
        times = []
        
        for i in range(iterations):
            try:
                start = time.time()
                response = requests.get(f"{self.api_url}{endpoint}", timeout=5)
                elapsed = time.time() - start
                
                if response.status_code == 200:
                    times.append(elapsed)
            
            except Exception as e:
                logger.warning(f"Request {i+1} failed: {e}")
        
        if not times:
            return {"error": "No successful requests"}
        
        results = {
            "endpoint": endpoint,
            "iterations": len(times),
            "avg_time_ms": round(sum(times) / len(times) * 1000, 2),
            "min_time_ms": round(min(times) * 1000, 2),
            "max_time_ms": round(max(times) * 1000, 2),
            "success_rate": f"{(len(times)/iterations)*100:.1f}%"
        }
        
        logger.info(f"Average response time: {results['avg_time_ms']}ms")
        logger.info(f"Min/Max: {results['min_time_ms']}ms / {results['max_time_ms']}ms")
        
        return results
    
    def generate_report(self) -> Dict:
        """Generate full health report."""
        logger.info("="*50)
        logger.info("Generating Health Report")
        logger.info("="*50)
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "api_url": self.api_url,
            "health_check": self.check_health(),
            "endpoints_test": self.test_endpoints(),
            "performance_test": self.performance_test()
        }
        
        # Save report
        with open("health_report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        logger.info("\n" + "="*50)
        logger.info("Report Summary:")
        logger.info("="*50)
        logger.info(f"Health: {'✓ Healthy' if report['health_check'] else '✗ Unhealthy'}")
        logger.info(f"Endpoints: {report['endpoints_test']['success']}/{report['endpoints_test']['total']} OK")
        logger.info(f"Avg Response Time: {report['performance_test'].get('avg_time_ms', 'N/A')}ms")
        logger.info("Report saved to health_report.json")
        
        return report

if __name__ == "__main__":
    import sys
    
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    
    monitor = HealthMonitor(base_url)
    monitor.generate_report()
