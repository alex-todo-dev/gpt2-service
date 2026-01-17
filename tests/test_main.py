
import json
from fastapi.testclient import TestClient
from ml_ops_exercise.main import app

client = TestClient(app)


class TestHealthRoute:
    """
    Route: /health
    Summary: availability check
    """

    def test_health_route_success(self):
        """
        Request: GET 
        Explanation: Verifies the API server is initialized
        Expectation: Status Code 200 OK and a JSON body containing {"status": "MAIN ROUTE API response"}.
        """
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "HEALTHY"}

class TestEncodeRoute:
    """
    Route: /encode
    Summary: converts text to embeddings
    """
    def test_encode_success(self):
        """
        Request: POST
        Explanation: Test of correct scenario 
        Expectation: Status Code 200 
        { "text": "That is my text" } -> {"tokens": [2504,318,616,2420],"count": 4}
        """
        response = client.post(
            "/encode",
            json={"text":"That is my text"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "tokens" in data
        assert "count" in data
        assert data['tokens'] == [2504,318,616,2420]
        assert data['count'] == len(data['tokens'])


    def test_empty_text(self):
        """
        Request: POST
        Explanation: Test of empty string input
        Expectation: Status Code 422,  "type": "string_too_short",
        """

        response = client.post(
            "/encode",
            json={"text":""}
        )
        assert response.status_code == 422


    def test_invalid_input_json(self):
        """
        Request: POST
        Explanation: Test of wrong key input ("data" <-> "text")
        Expectation: Status Code 422,  "type": "missing"
        """

        response = client.post(
            "/encode",
            json={"data":"That is my text"}
        )
        assert response.status_code == 422



class TestDecodeRoute:
    """
    Route: /decode
    Summary: converts embeddings to text
    """

    def test_decode_success(self):
        """
        Request: POST
        Explanation: Test of correct scenario. [2504,318,616,2420] -> {"text": "That is my text"}
        Expectation: Status Code 200,  {"text": "That is my text"}
        """
        response = client.post(
            "/decode",
            json={"tokens": [2504,318,616,2420]}
        )
        assert response.status_code == 200
        data = response.json()
        assert 'text' in data
        assert data['text'] == "That is my text"
    
    def test_long_list_input(self):
        """
        Request: POST
        Explanation: List of embeddings above the limit 1024
        Expectation: Status Code 422,   "msg": "List should have at most 1024 items after validation, ...",
        """
        response = client.post(
            "/decode",
            json={"tokens": [2504,318,616,2420]*2000}
        )
        assert response.status_code == 422

    def test_invalid_input_json(self):
        """
        Request: POST
        Explanation: Testing incorrect json {"tokens": [....]} <-> {"data": [....]}
        Expectation: Status Code 422,   "type": "missing"
        """
        response = client.post(
            "/decode",
            json={"data": [2504,318,616,2420]}
        )
        assert response.status_code == 422
    
    def test_empty_token_list(self):
        """
        Request: POST
        Explanation: Testing incorrect json {"tokens": []} 
        Expectation: Status Code 422,   "msg": "List should have at least 1 item after validation, not 0"
        """
        response = client.post(
            "/decode",
            json={"data": []}
        )
        assert response.status_code == 422


class TestGenerateRoute:
    """
    Route: /generate
    Summary: Given text input generates next words 
    """

    def test_generate_success(self):
        """
        Request: POST
        Explanation: Test of correct scenario. 
        Expectation: Status Code 200
        """

        input = { "prompt": "FAST API is","temp":0.2 ,"top_p": 0.95, "max_new_tokens": 10, "num_return_sequences": 1}

        response = client.post(
            "/generate",
            json=input
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data['result']) == 1

    def test_max_new_tokens(self):
        """
        Request: POST
        Explanation: Max generation tokens above the limit > 1024
        Expectation: Status Code 422 ,  "msg": "Input should be less than or equal to 1024"
        """

        input = { "prompt": "FAST API is","temp":0.2 ,"top_p": 0.95, "max_new_tokens": 1025, "num_return_sequences": 1}
        response = client.post(
            "/generate",
            json=input
        )

        assert response.status_code == 422

    def test_temperature(self):
        """
        Request: POST
        Explanation: Max generation tokens above the limit > 1024
        Expectation: Status Code 422 ,  "msg": "Input should be less than or equal to 1"
        """

        input = { "prompt": "FAST API is","temp":1.2 ,"top_p": 0.95, "max_new_tokens": 1025, "num_return_sequences": 1}
        response = client.post(
            "/generate",
            json=input
        )

        assert response.status_code == 422


    def test_top_p(self):
        """
        Request: POST
        Explanation: Top_p above the limit > 1
        Expectation: Status Code 422 ,  "msg": "Input should be less than or equal to 1"
        """

        input = { "prompt": "FAST API is","temp": 1 ,"top_p": 1.1, "max_new_tokens": 1025, "num_return_sequences": 1}
        response = client.post(
            "/generate",
            json=input
        )

        assert response.status_code == 422
    
    def test_empty_prompt(self):
        """
        Request: POST
        Explanation: Empty prompt value
        Expectation: Status Code 422 ,  "msg": "String should have at least 1 character"
        """
        input = { "prompt": "","temp": 1 ,"top_p": 1.1, "max_new_tokens": 1025, "num_return_sequences": 1}
        
        response = client.post(
            "/generate",
            json=input
        )

        assert response.status_code == 422


# poetry run pytest tests/test_main.py -v