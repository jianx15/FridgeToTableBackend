# myapp/views.py
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from . import model

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
