# Day 60: Advanced Testing - Complete Course

## Learning Objective
By the end of this lesson, you will be able to implement comprehensive testing strategies including integration testing, API testing, mocking techniques, and performance testing to ensure your applications are robust and production-ready.

---

## Introduction: The Master Chef's Quality Control

Imagine that you're running a five-star restaurant kitchen, and you've just been promoted from line cook to head chef. You've mastered individual cooking techniques (unit testing), but now you need to ensure that entire meals work together perfectly, that your suppliers deliver quality ingredients on time, and that your kitchen can handle the dinner rush without breaking down.

In the programming world, this is exactly what advanced testing does for your applications. Just as a master chef doesn't just taste individual components but ensures the entire dining experience is flawless, advanced testing goes beyond checking individual functions to verify that your entire application ecosystem works harmoniously.

---

## Lesson 1: Integration Testing - When Dishes Come Together

### The Kitchen Analogy
Think of integration testing like preparing a full course meal. You might have perfectly seasoned soup, a beautifully grilled steak, and exquisite dessert, but do they work together as a cohesive dining experience? Integration testing ensures that when your application's components work together, they create a seamless user experience.

### What is Integration Testing?
Integration testing verifies that different modules or services in your application work correctly when combined. It's the bridge between unit testing (testing individual functions) and end-to-end testing (testing the entire application flow).

### Code Example: Testing a User Registration System

```python
import pytest
import requests
from unittest.mock import patch, MagicMock
from myapp.models import User, Database
from myapp.services import EmailService, UserService
from myapp.validators import UserValidator

class TestUserRegistrationIntegration:
    """Integration tests for user registration flow"""
    
    def setup_method(self):
        """Set up test environment - like preparing mise en place"""
        self.db = Database(test_mode=True)
        self.email_service = EmailService(test_mode=True)
        self.user_service = UserService(self.db, self.email_service)
        
    def test_complete_user_registration_flow(self):
        """Test the entire user registration process"""
        # Arrange - prepare ingredients
        user_data = {
            'username': 'newchef',
            'email': 'chef@restaurant.com',
            'password': 'SecurePassword123!',
            'first_name': 'Gordon',
            'last_name': 'Ramsay'
        }
        
        # Act - execute the cooking process
        result = self.user_service.register_user(user_data)
        
        # Assert - taste test the final dish
        assert result.success is True
        assert result.user_id is not None
        
        # Verify user was saved to database
        saved_user = self.db.get_user_by_email(user_data['email'])
        assert saved_user is not None
        assert saved_user.username == user_data['username']
        assert saved_user.is_active is False  # Should be inactive until email verified
        
        # Verify welcome email was sent
        sent_emails = self.email_service.get_sent_emails()
        assert len(sent_emails) == 1
        assert sent_emails[0]['to'] == user_data['email']
        assert 'Welcome' in sent_emails[0]['subject']
        
    def test_registration_with_duplicate_email(self):
        """Test handling of duplicate email registration"""
        user_data = {
            'username': 'chef1',
            'email': 'existing@restaurant.com',
            'password': 'Password123!',
            'first_name': 'Julia',
            'last_name': 'Child'
        }
        
        # First registration should succeed
        result1 = self.user_service.register_user(user_data)
        assert result1.success is True
        
        # Second registration with same email should fail
        user_data['username'] = 'chef2'
        result2 = self.user_service.register_user(user_data)
        assert result2.success is False
        assert 'email already exists' in result2.error_message.lower()
        
    def teardown_method(self):
        """Clean up after tests - like cleaning the kitchen"""
        self.db.clear_test_data()
        self.email_service.clear_sent_emails()
```

### Syntax Explanation:
- `setup_method()` and `teardown_method()`: PyTest fixtures that run before and after each test
- `assert`: Python's built-in assertion statement for verifying expected outcomes
- `self.db = Database(test_mode=True)`: Creates a test database instance separate from production
- `result.success` and `result.user_id`: Accessing attributes of a custom result object

---

## Lesson 2: Testing with External APIs - Quality Control for Your Suppliers

### The Kitchen Analogy
Imagine your restaurant depends on various suppliers: the fishmonger delivers fresh salmon, the dairy farm provides cream, and the spice merchant supplies exotic seasonings. You need to ensure these suppliers deliver quality ingredients on time. Similarly, your application likely depends on external APIs for payment processing, weather data, social media integration, etc.

### What is API Testing?
API testing verifies that your application correctly communicates with external services, handles various response scenarios, and gracefully manages failures when those services are unavailable.

### Code Example: Testing Weather Service Integration

```python
import pytest
import requests
from unittest.mock import patch, Mock
from myapp.services import WeatherService, WeatherError

class TestWeatherAPIIntegration:
    """Test integration with external weather API"""
    
    def setup_method(self):
        self.weather_service = WeatherService(api_key="test_key")
        
    @patch('requests.get')
    def test_successful_weather_fetch(self, mock_get):
        """Test successful weather API response"""
        # Arrange - prepare mock response like a reliable supplier
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'location': {'name': 'New York'},
            'current': {
                'temperature': 72,
                'condition': 'Sunny',
                'humidity': 45
            }
        }
        mock_get.return_value = mock_response
        
        # Act - make the API call
        weather_data = self.weather_service.get_current_weather('New York')
        
        # Assert - verify the ingredients are properly processed
        assert weather_data['temperature'] == 72
        assert weather_data['condition'] == 'Sunny'
        assert weather_data['location'] == 'New York'
        
        # Verify the API was called correctly
        mock_get.assert_called_once_with(
            'https://api.weatherapi.com/v1/current.json',
            params={'key': 'test_key', 'q': 'New York'},
            timeout=10
        )
        
    @patch('requests.get')
    def test_api_timeout_handling(self, mock_get):
        """Test handling of API timeout - like a late supplier"""
        # Arrange - simulate a delayed delivery
        mock_get.side_effect = requests.Timeout("Connection timeout")
        
        # Act & Assert - ensure graceful failure handling
        with pytest.raises(WeatherError) as exc_info:
            self.weather_service.get_current_weather('Boston')
            
        assert 'timeout' in str(exc_info.value).lower()
        
    @patch('requests.get')
    def test_invalid_api_key(self, mock_get):
        """Test handling of authentication errors"""
        # Arrange - simulate rejected credentials
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.json.return_value = {'error': 'Invalid API key'}
        mock_get.return_value = mock_response
        
        # Act & Assert
        with pytest.raises(WeatherError) as exc_info:
            self.weather_service.get_current_weather('Chicago')
            
        assert 'authentication' in str(exc_info.value).lower()
        
    @patch('requests.get')
    def test_rate_limiting_handling(self, mock_get):
        """Test handling of rate limiting - like a supplier running out of stock"""
        # Arrange
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.headers = {'Retry-After': '60'}
        mock_get.return_value = mock_response
        
        # Act & Assert
        with pytest.raises(WeatherError) as exc_info:
            self.weather_service.get_current_weather('Miami')
            
        assert 'rate limit' in str(exc_info.value).lower()
```

### Syntax Explanation:
- `@patch('requests.get')`: Decorator that replaces the real requests.get with a mock during the test
- `mock_get.side_effect = requests.Timeout()`: Makes the mock raise an exception when called
- `pytest.raises(WeatherError)`: Context manager that verifies an exception is raised
- `mock_get.assert_called_once_with()`: Verifies the mock was called with specific parameters

---

## Lesson 3: Mock and Patch Techniques - Creating Stand-in Ingredients

### The Kitchen Analogy
Sometimes when developing a new recipe, you can't wait for the seasonal truffle delivery or the artisanal cheese to arrive. You need to create a "stand-in" ingredient that behaves similarly so you can test your cooking technique. In testing, mocks and patches serve the same purpose - they're stand-ins for real components that aren't available or are too expensive/slow to use in tests.

### What are Mocks and Patches?
- **Mock**: A fake object that simulates the behavior of real objects
- **Patch**: A technique to temporarily replace parts of your system during testing

### Code Example: Testing Payment Processing

```python
import pytest
from unittest.mock import Mock, patch, MagicMock
from decimal import Decimal
from myapp.services import PaymentService, OrderService
from myapp.models import Order, PaymentResult
from myapp.exceptions import PaymentError, InsufficientFundsError

class TestPaymentProcessing:
    """Test payment processing with various mocking techniques"""
    
    def setup_method(self):
        self.payment_service = PaymentService()
        self.order_service = OrderService()
        
    def test_successful_payment_with_mock_gateway(self):
        """Test successful payment using a mock payment gateway"""
        # Arrange - create a mock payment gateway (like a practice credit card machine)
        mock_payment_gateway = Mock()
        mock_payment_gateway.process_payment.return_value = PaymentResult(
            success=True,
            transaction_id='txn_123456',
            amount=Decimal('29.99'),
            fees=Decimal('0.87')
        )
        
        # Inject the mock into our service
        self.payment_service.gateway = mock_payment_gateway
        
        # Create test order
        order = Order(
            id='order_001',
            total=Decimal('29.99'),
            customer_email='customer@example.com'
        )
        
        # Act - process the payment
        result = self.payment_service.process_order_payment(order)
        
        # Assert - verify the payment was processed correctly
        assert result.success is True
        assert result.transaction_id == 'txn_123456'
        assert result.net_amount == Decimal('29.12')  # 29.99 - 0.87 fees
        
        # Verify the gateway was called with correct parameters
        mock_payment_gateway.process_payment.assert_called_once_with(
            amount=Decimal('29.99'),
            currency='USD',
            customer_email='customer@example.com'
        )
        
    @patch('myapp.services.EmailService.send_email')
    @patch('myapp.services.PaymentGateway.process_payment')
    def test_payment_failure_with_notification(self, mock_process, mock_send_email):
        """Test payment failure handling with email notification"""
        # Arrange - simulate a declined payment
        mock_process.side_effect = InsufficientFundsError("Card declined")
        mock_send_email.return_value = True
        
        order = Order(
            id='order_002',
            total=Decimal('149.99'),
            customer_email='customer@example.com'
        )
        
        # Act & Assert - ensure payment failure is handled gracefully
        with pytest.raises(PaymentError):
            self.payment_service.process_order_payment(order)
            
        # Verify failure notification was sent
        mock_send_email.assert_called_once()
        call_args = mock_send_email.call_args[1]  # Get keyword arguments
        assert call_args['to'] == 'customer@example.com'
        assert 'payment failed' in call_args['subject'].lower()
        
    def test_payment_with_partial_mock(self):
        """Test using MagicMock for complex object behavior"""
        # Arrange - create a sophisticated mock that can handle various scenarios
        mock_gateway = MagicMock()
        
        # Configure the mock to behave differently based on amount
        def mock_payment_logic(amount, currency='USD', **kwargs):
            if amount > 1000:
                raise PaymentError("Amount exceeds limit")
            elif amount < 1:
                raise PaymentError("Invalid amount")
            else:
                return PaymentResult(
                    success=True,
                    transaction_id=f'txn_{amount}_{currency}',
                    amount=amount,
                    fees=amount * Decimal('0.029')  # 2.9% fee
                )
        
        mock_gateway.process_payment.side_effect = mock_payment_logic
        self.payment_service.gateway = mock_gateway
        
        # Test various scenarios
        small_order = Order(id='small', total=Decimal('10.00'))
        result = self.payment_service.process_order_payment(small_order)
        assert result.success is True
        assert result.fees == Decimal('0.29')
        
        # Test amount limit
        large_order = Order(id='large', total=Decimal('1500.00'))
        with pytest.raises(PaymentError) as exc_info:
            self.payment_service.process_order_payment(large_order)
        assert 'exceeds limit' in str(exc_info.value)
```

### Advanced Mocking Techniques:

```python
class TestAdvancedMocking:
    """Demonstrate advanced mocking patterns"""
    
    @patch.object(PaymentService, 'validate_card')
    def test_method_specific_patching(self, mock_validate):
        """Patch a specific method on a class"""
        mock_validate.return_value = True
        
        service = PaymentService()
        result = service.validate_card('4111111111111111')
        
        assert result is True
        mock_validate.assert_called_once_with('4111111111111111')
        
    @patch('myapp.services.datetime')
    def test_datetime_mocking(self, mock_datetime):
        """Mock datetime for consistent time-based testing"""
        # Arrange - set a fixed time like synchronizing all kitchen clocks
        from datetime import datetime
        fixed_time = datetime(2024, 1, 15, 10, 30, 0)
        mock_datetime.now.return_value = fixed_time
        
        # Act
        service = PaymentService()
        timestamp = service.get_transaction_timestamp()
        
        # Assert
        assert timestamp == fixed_time
        
    def test_context_manager_mock(self):
        """Test mocking context managers like database connections"""
        with patch('myapp.database.DatabaseConnection') as mock_db:
            # Configure the context manager behavior
            mock_connection = Mock()
            mock_db.return_value.__enter__.return_value = mock_connection
            mock_connection.execute.return_value = [{'id': 1, 'name': 'Test'}]
            
            # Test the service that uses database
            service = PaymentService()
            result = service.get_payment_history('user_123')
            
            # Verify database interaction
            mock_db.assert_called_once()
            mock_connection.execute.assert_called()
```

### Syntax Explanation:
- `Mock()`: Creates a mock object that records how it's used
- `MagicMock()`: More powerful mock that can handle attribute access and method calls
- `mock.side_effect = exception`: Makes the mock raise an exception when called
- `mock.side_effect = function`: Makes the mock call a function when invoked
- `@patch.object(Class, 'method')`: Patches a specific method on a class
- `call_args[1]`: Accesses keyword arguments from the last call to a mock

---

## Lesson 4: Performance Testing - Ensuring Your Kitchen Can Handle the Rush

### The Kitchen Analogy
Your restaurant has been featured in a famous food magazine, and you're expecting a massive dinner rush. Will your kitchen be able to handle 200 orders instead of the usual 50? Will your prep stations become bottlenecks? Will your servers be able to keep up? Performance testing is like running a simulation of your busiest night to identify potential problems before they happen.

### What is Performance Testing?
Performance testing evaluates how your application behaves under various load conditions, measuring response times, throughput, and resource usage to ensure it can handle expected (and unexpected) user traffic.

### Code Example: Load Testing a Web API

```python
import pytest
import time
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor, as_completed
from statistics import mean, median
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class PerformanceResult:
    """Container for performance test results"""
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_response_time: float
    median_response_time: float
    min_response_time: float
    max_response_time: float
    requests_per_second: float
    error_rate: float

class PerformanceTester:
    """A performance testing toolkit for your applications"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.results = []
        
    def single_request_test(self, endpoint: str, method: str = 'GET', 
                           payload: dict = None) -> Dict[str, Any]:
        """Test a single request - like timing one dish preparation"""
        import requests
        
        start_time = time.time()
        try:
            if method == 'GET':
                response = requests.get(f"{self.base_url}{endpoint}", timeout=30)
            elif method == 'POST':
                response = requests.post(f"{self.base_url}{endpoint}", 
                                       json=payload, timeout=30)
            
            end_time = time.time()
            response_time = end_time - start_time
            
            return {
                'success': response.status_code < 400,
                'response_time': response_time,
                'status_code': response.status_code,
                'response_size': len(response.content)
            }
        except Exception as e:
            return {
                'success': False,
                'response_time': time.time() - start_time,
                'error': str(e)
            }
    
    def load_test(self, endpoint: str, concurrent_users: int = 10, 
                  requests_per_user: int = 10, method: str = 'GET',
                  payload: dict = None) -> PerformanceResult:
        """Simulate multiple customers ordering simultaneously"""
        
        def user_session():
            """Simulate a single user making multiple requests"""
            session_results = []
            for _ in range(requests_per_user):
                result = self.single_request_test(endpoint, method, payload)
                session_results.append(result)
                time.sleep(0.1)  # Brief pause between requests
            return session_results
        
        # Start the load test - like opening all kitchen stations at once
        print(f"Starting load test: {concurrent_users} users, "
              f"{requests_per_user} requests each")
        
        start_time = time.time()
        all_results = []
        
        # Use ThreadPoolExecutor to simulate concurrent users
        with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            futures = [executor.submit(user_session) for _ in range(concurrent_users)]
            
            for future in as_completed(futures):
                session_results = future.result()
                all_results.extend(session_results)
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        # Analyze results - like reviewing kitchen performance
        return self._analyze_results(all_results, total_duration)
    
    def _analyze_results(self, results: List[Dict], duration: float) -> PerformanceResult:
        """Analyze performance results like a head chef reviewing service"""
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]
        
        response_times = [r['response_time'] for r in successful]
        
        if not response_times:
            response_times = [0]  # Avoid division by zero
        
        return PerformanceResult(
            total_requests=len(results),
            successful_requests=len(successful),
            failed_requests=len(failed),
            average_response_time=mean(response_times),
            median_response_time=median(response_times),
            min_response_time=min(response_times),
            max_response_time=max(response_times),
            requests_per_second=len(results) / duration if duration > 0 else 0,
            error_rate=len(failed) / len(results) * 100 if results else 0
        )
    
    async def stress_test_async(self, endpoint: str, max_concurrent: int = 100,
                               duration_seconds: int = 60) -> PerformanceResult:
        """Async stress test - like testing kitchen during peak hours"""
        
        async def make_request(session, url):
            """Single async request"""
            start_time = time.time()
            try:
                async with session.get(url, timeout=30) as response:
                    await response.read()
                    return {
                        'success': response.status < 400,
                        'response_time': time.time() - start_time,
                        'status_code': response.status
                    }
            except Exception as e:
                return {
                    'success': False,
                    'response_time': time.time() - start_time,
                    'error': str(e)
                }
        
        # Run stress test
        end_time = time.time() + duration_seconds
        results = []
        
        async with aiohttp.ClientSession() as session:
            while time.time() < end_time:
                # Create batch of concurrent requests
                tasks = [
                    make_request(session, f"{self.base_url}{endpoint}")
                    for _ in range(max_concurrent)
                ]
                
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results
                for result in batch_results:
                    if isinstance(result, dict):
                        results.append(result)
                    else:
                        results.append({
                            'success': False,
                            'response_time': 0,
                            'error': str(result)
                        })
                
                # Brief pause before next batch
                await asyncio.sleep(0.1)
        
        return self._analyze_results(results, duration_seconds)

# Test usage examples
class TestPerformanceScenarios:
    """Real-world performance testing scenarios"""
    
    def setup_method(self):
        self.tester = PerformanceTester("http://localhost:8000")
    
    def test_api_endpoint_performance(self):
        """Test API endpoint under normal load"""
        result = self.tester.load_test(
            endpoint="/api/users",
            concurrent_users=5,
            requests_per_user=20
        )
        
        # Performance assertions - like quality standards
        assert result.error_rate < 1.0, f"Error rate too high: {result.error_rate}%"
        assert result.average_response_time < 2.0, \
            f"Average response time too slow: {result.average_response_time}s"
        assert result.requests_per_second > 10, \
            f"Throughput too low: {result.requests_per_second} RPS"
        
        print(f"Performance Results:")
        print(f"  Total Requests: {result.total_requests}")
        print(f"  Success Rate: {(result.successful_requests/result.total_requests)*100:.1f}%")
        print(f"  Average Response Time: {result.average_response_time:.3f}s")
        print(f"  Requests per Second: {result.requests_per_second:.1f}")
    
    def test_database_query_performance(self):
        """Test database-heavy endpoints"""
        # Test with various query complexities
        simple_query = self.tester.load_test("/api/users/simple", 10, 50)
        complex_query = self.tester.load_test("/api/reports/complex", 5, 10)
        
        # Verify performance degrades gracefully
        assert simple_query.average_response_time < complex_query.average_response_time
        assert simple_query.requests_per_second > complex_query.requests_per_second
    
    @pytest.mark.asyncio
    async def test_stress_conditions(self):
        """Test application under extreme load"""
        result = await self.tester.stress_test_async(
            endpoint="/api/health",
            max_concurrent=50,
            duration_seconds=30
        )
        
        # Even under stress, critical endpoints should remain responsive
        assert result.error_rate < 5.0, "Too many errors under stress"
        assert result.average_response_time < 5.0, "Response time too slow under stress"
    
    def test_memory_leak_detection(self):
        """Test for memory leaks during sustained load"""
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run sustained load test
        for i in range(5):
            result = self.tester.load_test("/api/users", 10, 20)
            current_memory = process.memory_info().rss / 1024 / 1024
            
            print(f"Iteration {i+1}: Memory usage {current_memory:.1f}MB")
            
            # Memory shouldn't grow indefinitely
            memory_growth = current_memory - initial_memory
            assert memory_growth < 100, f"Potential memory leak: {memory_growth:.1f}MB growth"
```

### Syntax Explanation:
- `@dataclass`: Python decorator that automatically generates `__init__`, `__repr__`, etc.
- `ThreadPoolExecutor`: Allows concurrent execution of functions in separate threads
- `as_completed(futures)`: Yields futures as they complete (not in submission order)
- `asyncio.gather(*tasks)`: Runs multiple async tasks concurrently
- `psutil.Process()`: Cross-platform library for system and process utilities
- `time.time()`: Returns current time as floating-point seconds since epoch

---

## Assignment: Building a Restaurant Review System Test Suite

### Project Description
You'll create a comprehensive test suite for a restaurant review system that demonstrates all four advanced testing techniques covered in this lesson.

### The System to Test
```python
# restaurant_system.py - The system you need to test
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime
import requests

@dataclass
class Review:
    id: str
    restaurant_id: str
    customer_email: str
    rating: int  # 1-5 stars
    comment: str
    created_at: datetime
    verified: bool = False

class RestaurantReviewSystem:
    def __init__(self, db_connection, notification_service, maps_api_key):
        self.db = db_connection
        self.notification_service = notification_service
        self.maps_api_key = maps_api_key
        
    def submit_review(self, review_data: dict) -> dict:
        """Submit a new restaurant review"""
        # Validate review data
        if not self._validate_review(review_data):
            return {"success": False, "error": "Invalid review data"}
        
        # Check if restaurant exists using Maps API
        restaurant_info = self._get_restaurant_info(review_data['restaurant_id'])
        if not restaurant_info:
            return {"success": False, "error": "Restaurant not found"}
        
        # Save review to database
        review = Review(
            id=f"rev_{datetime.now().timestamp()}",
            restaurant_id=review_data['restaurant_id'],
            customer_email=review_data['customer_email'],
            rating=review_data['rating'],
            comment=review_data['comment'],
            created_at=datetime.now()
        )
        
        self.db.save_review(review)
        
        # Send notification to restaurant owner
        self._notify_restaurant_owner(review, restaurant_info)
        
        return {"success": True, "review_id": review.id}
    
    def _validate_review(self, data: dict) -> bool:
        """Validate review data"""
        required_fields = ['restaurant_id', 'customer_email', 'rating', 'comment']
        return all(field in data for field in required_fields)
    
    def _get_restaurant_info(self, restaurant_id: str) -> Optional[dict]:
        """Get restaurant info from Maps API"""
        try:
            response = requests.get(
                f"https://maps.googleapis.com/maps/api/place/details/json",
                params={
                    'place_id': restaurant_id,
                    'key': self.maps_api_key
                },
                timeout=10
            )
            if response.status_code == 200:
                return response.json()
        except Exception:
            return None
    
    def _notify_restaurant_owner(self, review: Review, restaurant_info: dict):
        """Send notification to restaurant owner"""
        self.notification_service.send_notification(
            to=restaurant_info['owner_email'],
            subject=f"New {review.rating}-star review",
            body=f"New review: {review.comment}"
        )
```

### Your Assignment Task
Create a comprehensive test file called `test_restaurant_review_system.py` that includes:

1. **Integration Tests** (2-3 tests):
   - Test the complete review submission flow
   - Test review submission with invalid restaurant
   - Test the interaction between database and notification service

2. **API Testing** (2-3 tests):
   - Test successful Maps API response
   - Test Maps API timeout/failure handling
   - Test Maps API rate limiting

3. **Mocking Tests** (3-4 tests):
   - Mock the database connection
   - Mock the notification service
   - Mock the Maps API responses
   - Test datetime mocking for consistent timestamps

4. **Performance Tests** (2 tests):
   - Test response time under normal load (10 concurrent users)
   - Test system behavior under stress (50 concurrent requests)

### Requirements:
- Use all the testing techniques demonstrated in the lessons
- Include proper setup and teardown methods
- Add meaningful assertions that verify both success and failure cases
- Document your test methods with clear docstrings
- Include at least one test that combines multiple mocking techniques

### Submission Guidelines:
Submit your `test_restaurant_review_system.py` file with:
- All tests properly organized in classes
- Clear test method names that describe what they're testing
- Proper use of mocks, patches, and async testing where appropriate
- Performance thresholds that make sense for a review system
- Comments explaining complex mocking setups

### Success Criteria:
Your test suite should:
- Cover all four advanced testing techniques
- Be runnable

# Django Comprehensive Test Suite - Final Project

## Project: Restaurant Management System Test Suite

You'll build a comprehensive test suite for a restaurant management system that includes unit tests, integration tests, API testing, and performance testing.

### Project Structure
```
restaurant_project/
├── restaurant/
│   ├── __init__.py
│   ├── models.py
│   ├── views.py
│   ├── serializers.py
│   ├── urls.py
│   └── utils.py
├── tests/
│   ├── __init__.py
│   ├── test_models.py
│   ├── test_views.py
│   ├── test_integration.py
│   ├── test_api.py
│   ├── test_performance.py
│   └── conftest.py
└── requirements.txt
```

### Step 1: Create the Restaurant Models

**models.py**
```python
from django.db import models
from django.contrib.auth.models import User
from django.core.validators import MinValueValidator, MaxValueValidator
from decimal import Decimal

class Restaurant(models.Model):
    name = models.CharField(max_length=200)
    address = models.TextField()
    phone = models.CharField(max_length=20)
    email = models.EmailField()
    rating = models.DecimalField(
        max_digits=3, 
        decimal_places=2,
        validators=[MinValueValidator(0), MaxValueValidator(5)]
    )
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.name
    
    @property
    def total_orders_today(self):
        from django.utils import timezone
        today = timezone.now().date()
        return self.orders.filter(created_at__date=today).count()

class MenuItem(models.Model):
    CATEGORY_CHOICES = [
        ('appetizer', 'Appetizer'),
        ('main', 'Main Course'),
        ('dessert', 'Dessert'),
        ('beverage', 'Beverage'),
    ]
    
    restaurant = models.ForeignKey(Restaurant, on_delete=models.CASCADE, related_name='menu_items')
    name = models.CharField(max_length=200)
    description = models.TextField()
    price = models.DecimalField(max_digits=8, decimal_places=2)
    category = models.CharField(max_length=20, choices=CATEGORY_CHOICES)
    is_available = models.BooleanField(default=True)
    preparation_time = models.IntegerField(help_text="Time in minutes")
    
    def __str__(self):
        return f"{self.name} - ${self.price}"
    
    def apply_discount(self, percentage):
        """Apply discount and return new price"""
        discount_amount = self.price * Decimal(percentage) / 100
        return self.price - discount_amount

class Order(models.Model):
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('preparing', 'Preparing'),
        ('ready', 'Ready'),
        ('delivered', 'Delivered'),
        ('cancelled', 'Cancelled'),
    ]
    
    restaurant = models.ForeignKey(Restaurant, on_delete=models.CASCADE, related_name='orders')
    customer = models.ForeignKey(User, on_delete=models.CASCADE, related_name='orders')
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    total_amount = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"Order #{self.id} - {self.customer.username}"
    
    def calculate_total(self):
        """Calculate total from order items"""
        total = sum(item.subtotal for item in self.items.all())
        self.total_amount = total
        self.save()
        return total

class OrderItem(models.Model):
    order = models.ForeignKey(Order, on_delete=models.CASCADE, related_name='items')
    menu_item = models.ForeignKey(MenuItem, on_delete=models.CASCADE)
    quantity = models.PositiveIntegerField(validators=[MinValueValidator(1)])
    unit_price = models.DecimalField(max_digits=8, decimal_places=2)
    
    @property
    def subtotal(self):
        return self.quantity * self.unit_price
    
    def __str__(self):
        return f"{self.quantity}x {self.menu_item.name}"
```

### Step 2: Create Views and API Endpoints

**views.py**
```python
from django.shortcuts import get_object_or_404
from django.db.models import Avg, Count
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from .models import Restaurant, MenuItem, Order, OrderItem
from .serializers import RestaurantSerializer, MenuItemSerializer, OrderSerializer

class RestaurantViewSet(viewsets.ModelViewSet):
    queryset = Restaurant.objects.all()
    serializer_class = RestaurantSerializer
    
    @action(detail=True, methods=['get'])
    def menu(self, request, pk=None):
        restaurant = self.get_object()
        menu_items = restaurant.menu_items.filter(is_available=True)
        serializer = MenuItemSerializer(menu_items, many=True)
        return Response(serializer.data)
    
    @action(detail=True, methods=['get'])
    def stats(self, request, pk=None):
        restaurant = self.get_object()
        stats = {
            'total_orders': restaurant.orders.count(),
            'total_menu_items': restaurant.menu_items.count(),
            'average_order_value': restaurant.orders.aggregate(
                avg=Avg('total_amount')
            )['avg'] or 0,
            'orders_today': restaurant.total_orders_today
        }
        return Response(stats)

class OrderViewSet(viewsets.ModelViewSet):
    queryset = Order.objects.all()
    serializer_class = OrderSerializer
    
    def create(self, request):
        """Create order with items"""
        order_data = request.data
        items_data = order_data.pop('items', [])
        
        # Create order
        serializer = self.get_serializer(data=order_data)
        serializer.is_valid(raise_exception=True)
        order = serializer.save()
        
        # Create order items
        for item_data in items_data:
            menu_item = get_object_or_404(MenuItem, id=item_data['menu_item_id'])
            OrderItem.objects.create(
                order=order,
                menu_item=menu_item,
                quantity=item_data['quantity'],
                unit_price=menu_item.price
            )
        
        order.calculate_total()
        return Response(OrderSerializer(order).data, status=status.HTTP_201_CREATED)
    
    @action(detail=True, methods=['post'])
    def update_status(self, request, pk=None):
        order = self.get_object()
        new_status = request.data.get('status')
        
        if new_status not in dict(Order.STATUS_CHOICES):
            return Response(
                {'error': 'Invalid status'}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        order.status = new_status
        order.save()
        
        return Response({'status': order.status})
```

### Step 3: Comprehensive Test Suite

**tests/conftest.py**
```python
import pytest
from django.contrib.auth.models import User
from restaurant.models import Restaurant, MenuItem, Order, OrderItem
from decimal import Decimal

@pytest.fixture
def user():
    return User.objects.create_user(
        username='testuser',
        email='test@example.com',
        password='testpass123'
    )

@pytest.fixture
def restaurant():
    return Restaurant.objects.create(
        name="Chef's Kitchen",
        address="123 Food Street",
        phone="+1234567890",
        email="info@chefskitchen.com",
        rating=Decimal('4.5')
    )

@pytest.fixture
def menu_items(restaurant):
    items = []
    items.append(MenuItem.objects.create(
        restaurant=restaurant,
        name="Grilled Salmon",
        description="Fresh Atlantic salmon",
        price=Decimal('24.99'),
        category='main',
        preparation_time=20
    ))
    items.append(MenuItem.objects.create(
        restaurant=restaurant,
        name="Caesar Salad",
        description="Crispy romaine lettuce",
        price=Decimal('12.99'),
        category='appetizer',
        preparation_time=10
    ))
    return items

@pytest.fixture
def sample_order(restaurant, user, menu_items):
    order = Order.objects.create(
        restaurant=restaurant,
        customer=user,
        status='pending'
    )
    
    # Add items to order
    for menu_item in menu_items:
        OrderItem.objects.create(
            order=order,
            menu_item=menu_item,
            quantity=1,
            unit_price=menu_item.price
        )
    
    order.calculate_total()
    return order
```

**tests/test_models.py**
```python
import pytest
from decimal import Decimal
from django.core.exceptions import ValidationError
from restaurant.models import Restaurant, MenuItem, Order, OrderItem

@pytest.mark.django_db
class TestRestaurantModel:
    
    def test_restaurant_creation(self):
        restaurant = Restaurant.objects.create(
            name="Test Restaurant",
            address="123 Test St",
            phone="555-0123",
            email="test@restaurant.com",
            rating=Decimal('4.2')
        )
        assert restaurant.name == "Test Restaurant"
        assert restaurant.rating == Decimal('4.2')
        assert str(restaurant) == "Test Restaurant"
    
    def test_restaurant_rating_validation(self):
        with pytest.raises(ValidationError):
            restaurant = Restaurant(
                name="Test Restaurant",
                address="123 Test St",
                phone="555-0123",
                email="test@restaurant.com",
                rating=Decimal('6.0')  # Invalid rating > 5
            )
            restaurant.full_clean()
    
    def test_total_orders_today_property(self, restaurant, sample_order):
        assert restaurant.total_orders_today == 1

@pytest.mark.django_db
class TestMenuItemModel:
    
    def test_menu_item_creation(self, restaurant):
        menu_item = MenuItem.objects.create(
            restaurant=restaurant,
            name="Test Dish",
            description="Test description",
            price=Decimal('15.99'),
            category='main',
            preparation_time=25
        )
        assert menu_item.name == "Test Dish"
        assert menu_item.price == Decimal('15.99')
    
    def test_apply_discount(self, menu_items):
        menu_item = menu_items[0]  # Grilled Salmon - $24.99
        discounted_price = menu_item.apply_discount(20)  # 20% discount
        expected_price = Decimal('19.99')
        assert discounted_price == expected_price

@pytest.mark.django_db
class TestOrderModel:
    
    def test_order_creation(self, restaurant, user):
        order = Order.objects.create(
            restaurant=restaurant,
            customer=user,
            status='pending'
        )
        assert order.customer == user
        assert order.status == 'pending'
        assert order.total_amount == 0
    
    def test_calculate_total(self, sample_order):
        # Sample order has 2 items: $24.99 + $12.99 = $37.98
        total = sample_order.calculate_total()
        expected_total = Decimal('37.98')
        assert total == expected_total
        assert sample_order.total_amount == expected_total
```

**tests/test_integration.py**
```python
import pytest
from django.test import TransactionTestCase
from django.db import transaction
from restaurant.models import Restaurant, MenuItem, Order, OrderItem
from django.contrib.auth.models import User
from decimal import Decimal

@pytest.mark.django_db
class TestOrderWorkflow:
    
    def test_complete_order_workflow(self, restaurant, user, menu_items):
        """Test complete order creation and status updates"""
        
        # Step 1: Create order
        order = Order.objects.create(
            restaurant=restaurant,
            customer=user,
            status='pending'
        )
        
        # Step 2: Add items
        for menu_item in menu_items:
            OrderItem.objects.create(
                order=order,
                menu_item=menu_item,
                quantity=2,
                unit_price=menu_item.price
            )
        
        # Step 3: Calculate total
        total = order.calculate_total()
        expected_total = (Decimal('24.99') + Decimal('12.99')) * 2
        assert total == expected_total
        
        # Step 4: Update status through workflow
        statuses = ['pending', 'preparing', 'ready', 'delivered']
        for status in statuses:
            order.status = status
            order.save()
            order.refresh_from_db()
            assert order.status == status
    
    def test_restaurant_stats_integration(self, restaurant, user, menu_items):
        """Test restaurant statistics calculation"""
        
        # Create multiple orders
        orders = []
        for i in range(3):
            order = Order.objects.create(
                restaurant=restaurant,
                customer=user,
                status='delivered',
                total_amount=Decimal('50.00') + (i * 10)
            )
            orders.append(order)
        
        # Test stats calculation
        total_orders = restaurant.orders.count()
        assert total_orders == 3
        
        # Calculate average manually to compare
        total_amount = sum(order.total_amount for order in orders)
        expected_avg = total_amount / len(orders)
        
        from django.db.models import Avg
        actual_avg = restaurant.orders.aggregate(avg=Avg('total_amount'))['avg']
        assert actual_avg == expected_avg

class TestDatabaseTransactions(TransactionTestCase):
    
    def test_order_creation_atomicity(self):
        """Test that order creation is atomic"""
        restaurant = Restaurant.objects.create(
            name="Test Restaurant",
            address="123 Test St",
            phone="555-0123",
            email="test@restaurant.com",
            rating=Decimal('4.0')
        )
        
        user = User.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpass123'
        )
        
        menu_item = MenuItem.objects.create(
            restaurant=restaurant,
            name="Test Item",
            description="Test",
            price=Decimal('10.00'),
            category='main',
            preparation_time=15
        )
        
        try:
            with transaction.atomic():
                order = Order.objects.create(
                    restaurant=restaurant,
                    customer=user,
                    status='pending'
                )
                
                # This should succeed
                OrderItem.objects.create(
                    order=order,
                    menu_item=menu_item,
                    quantity=1,
                    unit_price=menu_item.price
                )
                
                # Force an error to test rollback
                raise ValueError("Simulated error")
                
        except ValueError:
            pass  # Expected error
        
        # Check that no orders were created due to rollback
        assert Order.objects.count() == 0
        assert OrderItem.objects.count() == 0
```

**tests/test_api.py**
```python
import pytest
from django.urls import reverse
from rest_framework.test import APIClient
from rest_framework import status
from decimal import Decimal
import json

@pytest.mark.django_db
class TestRestaurantAPI:
    
    def test_restaurant_list(self, restaurant):
        client = APIClient()
        url = reverse('restaurant-list')
        response = client.get(url)
        
        assert response.status_code == status.HTTP_200_OK
        assert len(response.data) == 1
        assert response.data[0]['name'] == restaurant.name
    
    def test_restaurant_menu_endpoint(self, restaurant, menu_items):
        client = APIClient()
        url = reverse('restaurant-menu', kwargs={'pk': restaurant.id})
        response = client.get(url)
        
        assert response.status_code == status.HTTP_200_OK
        assert len(response.data) == 2  # 2 menu items from fixture
        
        # Check that only available items are returned
        menu_items[0].is_available = False
        menu_items[0].save()
        
        response = client.get(url)
        assert len(response.data) == 1
    
    def test_restaurant_stats_endpoint(self, restaurant, sample_order):
        client = APIClient()
        url = reverse('restaurant-stats', kwargs={'pk': restaurant.id})
        response = client.get(url)
        
        assert response.status_code == status.HTTP_200_OK
        stats = response.data
        assert stats['total_orders'] == 1
        assert stats['total_menu_items'] == 2
        assert Decimal(str(stats['average_order_value'])) == sample_order.total_amount

@pytest.mark.django_db
class TestOrderAPI:
    
    def test_create_order_with_items(self, restaurant, user, menu_items):
        client = APIClient()
        client.force_authenticate(user=user)
        
        url = reverse('order-list')
        order_data = {
            'restaurant': restaurant.id,
            'customer': user.id,
            'items': [
                {
                    'menu_item_id': menu_items[0].id,
                    'quantity': 2
                },
                {
                    'menu_item_id': menu_items[1].id,
                    'quantity': 1
                }
            ]
        }
        
        response = client.post(url, order_data, format='json')
        
        assert response.status_code == status.HTTP_201_CREATED
        
        # Verify order was created with correct total
        order_id = response.data['id']
        from restaurant.models import Order
        order = Order.objects.get(id=order_id)
        
        expected_total = (menu_items[0].price * 2) + menu_items[1].price
        assert order.total_amount == expected_total
        assert order.items.count() == 2
    
    def test_update_order_status(self, sample_order, user):
        client = APIClient()
        client.force_authenticate(user=user)
        
        url = reverse('order-update-status', kwargs={'pk': sample_order.id})
        response = client.post(url, {'status': 'preparing'})
        
        assert response.status_code == status.HTTP_200_OK
        assert response.data['status'] == 'preparing'
        
        sample_order.refresh_from_db()
        assert sample_order.status == 'preparing'
    
    def test_invalid_status_update(self, sample_order, user):
        client = APIClient()
        client.force_authenticate(user=user)
        
        url = reverse('order-update-status', kwargs={'pk': sample_order.id})
        response = client.post(url, {'status': 'invalid_status'})
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert 'error' in response.data
```

**tests/test_performance.py**
```python
import pytest
import time
from django.test import override_settings
from django.db import connection
from django.test.utils import override_settings
from restaurant.models import Restaurant, MenuItem, Order, OrderItem
from django.contrib.auth.models import User
from decimal import Decimal

@pytest.mark.django_db
class TestPerformance:
    
    def test_restaurant_query_optimization(self):
        """Test that restaurant queries are optimized"""
        
        # Create test data
        restaurants = []
        for i in range(10):
            restaurant = Restaurant.objects.create(
                name=f"Restaurant {i}",
                address=f"Address {i}",
                phone=f"555-010{i}",
                email=f"restaurant{i}@test.com",
                rating=Decimal('4.0')
            )
            restaurants.append(restaurant)
            
            # Add menu items to each restaurant
            for j in range(5):
                MenuItem.objects.create(
                    restaurant=restaurant,
                    name=f"Item {j}",
                    description="Test item",
                    price=Decimal('10.00'),
                    category='main',
                    preparation_time=15
                )
        
        # Test query count for getting restaurants with menu items
        with override_settings(DEBUG=True):
            start_queries = len(connection.queries)
            
            # This should use select_related/prefetch_related for optimization
            restaurants_with_items = Restaurant.objects.prefetch_related('menu_items').all()
            
            # Access the related data
            for restaurant in restaurants_with_items:
                list(restaurant.menu_items.all())  # Force evaluation
            
            end_queries = len(connection.queries)
            query_count = end_queries - start_queries
            
            # Should be much less than 10 restaurants * 5 menu items = 50 queries
            # With proper optimization, should be around 2 queries
            assert query_count <= 3
    
    def test_order_creation_performance(self, restaurant, user):
        """Test order creation performance with many items"""
        
        # Create menu items
        menu_items = []
        for i in range(20):
            menu_item = MenuItem.objects.create(
                restaurant=restaurant,
                name=f"Item {i}",
                description="Test item",
                price=Decimal('15.00'),
                category='main',
                preparation_time=20
            )
            menu_items.append(menu_item)
        
        # Time the order creation
        start_time = time.time()
        
        order = Order.objects.create(
            restaurant=restaurant,
            customer=user,
            status='pending'
        )
        
        # Add all items to order
        order_items = []
        for menu_item in menu_items:
            order_item = OrderItem(
                order=order,
                menu_item=menu_item,
                quantity=1,
                unit_price=menu_item.price
            )
            order_items.append(order_item)
        
        # Bulk create for better performance
        OrderItem.objects.bulk_create(order_items)
        
        order.calculate_total()
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Should complete within reasonable time (adjust threshold as needed)
        assert execution_time < 1.0  # Less than 1 second
        
        # Verify correctness
        assert order.items.count() == 20
        expected_total = Decimal('15.00') * 20
        assert order.total_amount == expected_total
    
    def test_database_query_time(self, restaurant, user, menu_items):
        """Test that database queries execute within acceptable time"""
        
        # Create multiple orders for testing
        orders = []
        for i in range(50):
            order = Order.objects.create(
                restaurant=restaurant,
                customer=user,
                status='delivered',
                total_amount=Decimal('25.00') + i
            )
            orders.append(order)
        
        # Test complex query performance
        start_time = time.time()
        
        from django.db.models import Avg, Count, Sum
        stats = Restaurant.objects.filter(id=restaurant.id).aggregate(
            total_orders=Count('orders'),
            avg_order_value=Avg('orders__total_amount'),
            total_revenue=Sum('orders__total_amount'),
            menu_items_count=Count('menu_items')
        )
        
        end_time = time.time()
        query_time = end_time - start_time
        
        # Query should complete quickly
        assert query_time < 0.5  # Less than 500ms
        
        # Verify results
        assert stats['total_orders'] == 51  # 50 + 1 from fixture
        assert stats['menu_items_count'] == 2  # From fixture
```

### Step 4: Test Configuration Files

**pytest.ini**
```ini
[tool:pytest]
DJANGO_SETTINGS_MODULE = restaurant_project.settings
python_files = tests.py test_*.py *_tests.py
addopts = -v --tb=short --strict-markers
markers =
    slow: marks tests as slow
    integration: marks tests as integration tests
    api: marks tests as API tests
    performance: marks tests as performance tests
```

**requirements.txt**
```
Django>=4.2.0
djangorestframework>=3.14.0
pytest>=7.0.0
pytest-django>=4.5.0
pytest-cov>=4.0.0
factory-boy>=3.2.0
```

### Step 5: Running the Test Suite

**Run all tests:**
```bash
pytest tests/
```

**Run specific test categories:**
```bash
pytest tests/test_models.py -v
pytest tests/test_integration.py -v
pytest tests/test_api.py -v
pytest tests/test_performance.py -v
```

**Run with coverage:**
```bash
pytest tests/ --cov=restaurant --cov-report=html
```

**Run performance tests only:**
```bash
pytest tests/test_performance.py -v -m performance
```

### Assignment

**Assignment: Add Advanced Test Scenarios**

Extend the test suite by adding the following test files:

1. **tests/test_edge_cases.py** - Create tests for:
   - Order cancellation with refund calculations
   - Menu item price changes affecting existing orders
   - Restaurant capacity limits and overbooking scenarios
   - Invalid data handling (negative prices, invalid emails)

2. **tests/test_business_logic.py** - Test complex business rules:
   - Discount calculations for bulk orders (>5 items get 10% off)
   - Peak hours pricing (20% markup between 6-8 PM)
   - Loyalty points system (1 point per dollar spent)
   - Inventory tracking (reduce available items when ordered)

3. **Mock external payment processing API** in your tests using pytest fixtures and the `unittest.mock` library.

**Requirements:**
- Use proper test fixtures and setup/teardown
- Include both positive and negative test cases
- Add performance benchmarks for the new features
- Achieve at least 95% code coverage
- Document any complex test scenarios with docstrings

**Submission:**
- Submit the complete test files
- Include a coverage report
- Add a brief README explaining your testing strategy
- Show test execution results with timing information