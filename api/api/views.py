# myapp/views.py
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from . import model
from . import tf_model

@csrf_exempt
def predict_view(request):
    if request.method == 'POST':
        try:
            # Get the raw JSON data from the request body
            data = json.loads(request.body.decode('utf-8'))

            # Access the "data" key from the JSON
            input_data = data.get('data')
            print(input_data)
            result = model.Model().get_recommendation(input_data)
            return JsonResponse({'prediction': result})

        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON data'})

    return JsonResponse({'error': 'Invalid request method'})

@csrf_exempt
def recommend_view(request):
    if request.method == 'POST':
        try:
            # Get the raw JSON data from the request body
            data = json.loads(request.body.decode('utf-8'))

            # Access the "data" key from the JSON
            input_data_user_id = data.get('userId')
            input_data_food_type = data.get('foodType')
            input_data_ingredients = data.get('ingredients')

            result = tf_model.TFModel().recommend(input_data_user_id, input_data_food_type, input_data_ingredients)

            return JsonResponse({'prediction': str(round(result, 2))})

        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON data'})    

    return JsonResponse({'error': 'Invalid request method'})
