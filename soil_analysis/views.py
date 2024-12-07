import os
import requests
import joblib
import numpy as np
import cv2
import json
import re
from rest_framework.decorators import api_view
from rest_framework.response import Response
from keras.models import load_model
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
import google.generativeai as genai

# Load trained models
model1 = joblib.load("C:/Users/vijay_jjyhjd9/Desktop/final_django_npk/npk_predict/soil_analysis/npk_ph_predictor_model.pkl")  # NPK prediction model
soil_classifier_model = load_model("C:/Users/vijay_jjyhjd9/Desktop/final_django_npk/npk_predict/soil_analysis/keras_model.h5", compile=False)  # Soil classification model

# Configure Gemini AI
genai.configure(api_key="AIzaSyDf6tv5q58fpkRb5aH27UqTiLH8T7ehvw4")

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)
chat_session = model.start_chat()


# Feature extraction function
def extract_features_from_image(image):
    image = cv2.resize(image, (224, 224))
    mean_color = np.mean(image, axis=(0, 1))
    std_color = np.std(image, axis=(0, 1))
    sum_rgb = np.sum(mean_color)
    norm_r = mean_color[2] / sum_rgb
    norm_g = mean_color[1] / sum_rgb
    norm_b = mean_color[0] / sum_rgb
    green_dominance = mean_color[1] > (mean_color[0] + mean_color[2]) / 2
    blue_ratio = mean_color[0] / (mean_color[1] + mean_color[2])
    yellowish_blue_ratio = ((mean_color[2] + mean_color[1]) / 2) / mean_color[0]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), density=True)
    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    glcm_contrast = graycoprops(glcm, "contrast")[0, 0]
    glcm_homogeneity = graycoprops(glcm, "homogeneity")[0, 0]
    glcm_entropy = -np.sum(glcm * np.log2(glcm + (glcm == 0)))

    features = [
        mean_color[2], mean_color[1], mean_color[0],
        std_color[2], std_color[1], std_color[0],
        norm_r, norm_g, norm_b,
        green_dominance, blue_ratio, yellowish_blue_ratio,
        glcm_contrast, glcm_homogeneity, glcm_entropy
    ]
    return features


def classify_soil(image):
    image_resized = cv2.resize(image, (224, 224))
    image_array = np.expand_dims(image_resized, axis=0)
    prediction = soil_classifier_model.predict(image_array)
    confidence = np.max(prediction)
    predicted_class = np.argmax(prediction)
    return predicted_class, confidence


def parse_gemini_response(response_text):
    try:
        json_match = re.search(r'{.*}', response_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        else:
            raise ValueError("No JSON found in Gemini AI response")
    except json.JSONDecodeError:
        return {"error": "Invalid JSON format received from Gemini AI"}
    except Exception as e:
        return {"error": str(e)}


def get_fertilizer_recommendation(n_value, p_value, k_value, ph_value, crop_type, soil_type, weather):
    prompt = f"""
        Given the following soil and crop information, generate a **fertilizer recommendation** in **only JSON format**:
        
        - Nitrogen (N): {n_value}%, Phosphorus (P): {p_value} ppm, Potassium (K): {k_value} ppm, pH: {ph_value}
        - Crop Type: {crop_type}, Soil Type: {soil_type}, Weather: {weather}

        Provide JSON:
        {{
            "fertilizer_name": "...",
            "fertilizer_quantity": "...",
            "application_schedule": "...",
            "application_method": "...",
            "data": "..."
        }}
    """
    response = chat_session.send_message(prompt)
    return parse_gemini_response(response.text if response and hasattr(response, 'text') else "")


@api_view(['POST'])
def predict_npk(request):
    try:
        image_urls = request.data.get("image_urls", [])
        if not image_urls:
            return Response({"error": "No image URLs provided."}, status=400)

        all_features = []
        class_confidences = []
        for url in image_urls:
            try:
                response = requests.get(url)
                response.raise_for_status()
                img_array = np.array(bytearray(response.content), dtype=np.uint8)
                image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                predicted_class, confidence = classify_soil(image)
                class_confidences.append((predicted_class, confidence))
                if confidence >= 0.40:
                    features = extract_features_from_image(image)
                    all_features.append(features)
            except Exception as e:
                return Response({"error": f"Error processing image: {str(e)}"}, status=400)

        if not class_confidences or max(class_confidences, key=lambda x: x[1])[1] < 0.40:
            return Response({"error": "Low soil classification confidence."}, status=400)

        avg_features = np.mean(all_features, axis=0) if all_features else None
        if avg_features is not None:
            prediction = model1.predict([avg_features])
            n_value, p_value, k_value, ph_value = prediction[0]
            soil_types = {
                0: "Alluvial soil",
                1: "Black soil",
                2: "Chalky soil",
                3: "Clay soil",
                4: "Mary soil",
                5: "Red soil",
                6: "Sand soil",
                7: "Silt soil"
            }
            return Response({
                "n_value(%)": n_value,
                "p_value(ppm)": p_value,
                "k_value(ppm)": k_value,
                "ph_value": ph_value,
                "Predicted_Soil_Class": soil_types[max(class_confidences, key=lambda x: x[1])[0]],
                "Success": True
            })
        return Response({"error": "No valid images for NPK prediction."}, status=400)
    except Exception as e:
        return Response({"error": str(e)}, status=500)


@api_view(['POST'])
def recommend_fertilizer(request):
    try:
        data = request.data
        n_value = data.get("n_value")
        p_value = data.get("p_value")
        k_value = data.get("k_value")
        ph_value = data.get("ph_value")
        crop_type = data.get("crop_type", "wheat")
        soil_type = data.get("soil_type", "Alluvial Soil")
        weather = data.get("weather", "Humid")

        if None in [n_value, p_value, k_value, ph_value]:
            return Response({"error": "N, P, K, and pH values must be provided."}, status=400)

        recommendation = get_fertilizer_recommendation(n_value, p_value, k_value, ph_value, crop_type, soil_type, weather)
        if "fertilizer_name" in recommendation:
            return Response({"fertilizer_recommendation": recommendation})
        return Response({"error": "Failed to generate recommendation."}, status=500)
    except Exception as e:
        return Response({"error": str(e)}, status=500)
