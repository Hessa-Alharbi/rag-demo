const fetch = require('node-fetch');

// Configuration
const API_URL = 'https://ingenious-transformation-production-be7c.up.railway.app'; // Change to your backend URL
const TEST_USER = 'team@gmail.com'; // Change to your test user
const TEST_PASSWORD = 'password123'; // Change to your test password

async function testLogin() {
  console.log('Testing login functionality...');
  
  // Create form data
  const formData = new URLSearchParams();
  formData.append('username', TEST_USER);
  formData.append('password', TEST_PASSWORD);
  
  try {
    console.log(`Attempting login for user: ${TEST_USER}`);
    
    // Make the login request
    const response = await fetch(`${API_URL}/auth/login`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      body: formData.toString(),
    });
    
    // Get response data
    const responseData = await response.json().catch(() => null);
    
    if (response.ok) {
      console.log('Login successful!');
      console.log('Response:', responseData);
      
      // Test the /me endpoint with the token
      if (responseData && responseData.access_token) {
        console.log('\nTesting /me endpoint with the token...');
        const meResponse = await fetch(`${API_URL}/auth/me`, {
          headers: {
            'Authorization': `Bearer ${responseData.access_token}`
          }
        });
        
        const userData = await meResponse.json().catch(() => null);
        
        if (meResponse.ok) {
          console.log('User data retrieved successfully:');
          console.log(userData);
        } else {
          console.error('Failed to get user data:', meResponse.status, userData);
        }
      }
    } else {
      console.error('Login failed with status:', response.status);
      console.error('Error:', responseData);
    }
  } catch (error) {
    console.error('Request error:', error.message);
  }
}

// Run the test
testLogin(); 
