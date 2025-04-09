from flask import Flask, render_template, request, redirect, url_for, session, flash
import mysql.connector
import pandas as pd
import random
from sklearn.cluster import KMeans
import os
from flask import session
from flask import session, redirect, url_for
from flask import Flask, render_template, Response, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import re
import sys
import unicodedata
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.saving import register_keras_serializable
import tensorflow.keras.losses as losses
import cv2
import mediapipe as mp
import numpy as np
import logging

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'
# Load the posture analysis model
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()
# Register the custom loss function
@register_keras_serializable()
def custom_loss(y_true, y_pred):
    return losses.mean_squared_error(y_true, y_pred)  # Replace with your actual loss function

# Load the clustered dataset
file_path = os.path.join(os.path.dirname(__file__), 'data', 'Clustered_Nutrition.csv')
nutrition_data = pd.read_csv(file_path, encoding='utf-8')

# Load exercise data
exercise_data_path = os.path.join(os.path.dirname(__file__), 'data', 'gym recommendation.xlsx')
exercise_data = pd.read_excel(exercise_data_path, sheet_name='Sheet1', engine='openpyxl')

sys.stdout.reconfigure(encoding='utf-8')


def perform_kmeans_clustering(nutrition_data, n_clusters=5):
    features = nutrition_data[['energy_kcal', 'fat_g', 'protein_g', 'carb_g']]
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    nutrition_data['Cluster'] = kmeans.fit_predict(features)

perform_kmeans_clustering(nutrition_data, n_clusters=5)

# Manual Database Connection (using mysql.connector)
def get_db_connection():
    try:
        connection = mysql.connector.connect(
            host="localhost",
            user="root",
            password="mf90zk@ash123",
            database="nutrition_fitness_v2"
        )
        if connection.is_connected():
            return connection
    except mysql.connector.Error as e:
        print(f"Error: {e}")
        return None

# Helper Functions

def calculate_calories(age, height, weight, gender):
    bmr = (10 * weight + 6.25 * height - 5 * age + (5 if gender == 'male' else -161))
    return bmr * 1.55

def adjust_calories_for_goal(daily_calories, weight_goal):
    if weight_goal == 'lose':
        return daily_calories * 0.85
    elif weight_goal == 'gain':
        return daily_calories * 1.15
    return daily_calories

def filter_meal_type_and_conditions(data, meal_type, hypertension, diabetes):
    filtered = data[data['Meal Type'].str.lower() == meal_type.lower()]
    if hypertension:
        filtered = filtered[~filtered['ingredients'].str.contains('sodium|salt', case=False, na=False)]
    if diabetes:
        filtered = filtered[~filtered['ingredients'].str.contains('sugar|sweet', case=False, na=False)]
    return filtered

def select_item(data):
    return data.sample(1).iloc[0] if not data.empty else None

def select_personalized_exercise(gender, weight_goal, hypertension, diabetes):
    # Convert input parameters to lowercase for consistency
    gender = gender.lower()
    weight_goal = weight_goal.lower()
    
    # Filter based on gender and fitness goal
    conditions = (exercise_data['Sex'].str.lower() == gender) & \
                 (exercise_data['Fitness Goal'].str.lower() == weight_goal)

    # Further filtering based on health conditions
    if hypertension:
        conditions &= (exercise_data['Hypertension'].str.lower() == 'yes')
    if diabetes:
        conditions &= (exercise_data['Diabetes'].str.lower() == 'yes')

    # Select a personalized exercise
    filtered = exercise_data[conditions]
    if not filtered.empty:
        return filtered.sample(1)['Exercises'].values[0]  # Return a random exercise from the filtered list

    # If no specific match, return a more varied fallback option
    fallback = exercise_data.sample(1)['Exercises'].values[0]
    return fallback
# Routes



def detect_posture(frame, exercise):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        if exercise == "squat":
            return check_squat(landmarks)
        elif exercise == "plank":
            return check_plank(landmarks)
        elif exercise == "walking":
            return check_walking(landmarks)
    return "Incorrect"


def check_squat(landmarks):
    knee_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_HIP], 
                                landmarks[mp_pose.PoseLandmark.LEFT_KNEE], 
                                landmarks[mp_pose.PoseLandmark.LEFT_ANKLE])
    if 80 <= knee_angle <= 100:
        return "Correct"
    return "Incorrect"


def check_plank(landmarks):
    back_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER], 
                                 landmarks[mp_pose.PoseLandmark.LEFT_HIP], 
                                 landmarks[mp_pose.PoseLandmark.LEFT_KNEE])
    if 160 <= back_angle <= 180:
        return "Correct"
    return "Incorrect"



def check_walking(landmarks):
    # Calculate back angle (shoulder-hip-knee)
    back_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER], 
                                 landmarks[mp_pose.PoseLandmark.LEFT_HIP], 
                                 landmarks[mp_pose.PoseLandmark.LEFT_KNEE])


    # Define stricter thresholds
    BACK_ANGLE_CORRECT = (176, 180)  # Should be near perfect straight
    BACK_ANGLE_WARN = (160, 174)  # Slight bend

    # Check back posture
    if back_angle < BACK_ANGLE_WARN[0]:
        return "Incorrect - Back is bent too much (Leaning forward too much)"
    elif back_angle < BACK_ANGLE_CORRECT[0]:
        return " Incorrect - Back is slightly bent"


    return "Correct"


def calculate_angle(a, b, c):
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)


def generate_frames(exercise):
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        posture_status = detect_posture(frame, exercise)
        cv2.putText(frame, posture_status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()



@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # Using manual DB connection to check user credentials in the main_user table
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor(dictionary=True)
            cursor.execute('SELECT * FROM main_user WHERE username = %s', (username,))
            user = cursor.fetchone()
            cursor.close()
            conn.close()
            
            if user and user['password'] == password:
                session['user_id'] = user['user_id']  # Storing user_id in session
                return redirect(url_for('home'))
            else:
                flash('Invalid login credentials', 'danger')
        else:
            flash('Database connection failed', 'danger')
    
    return render_template('login.html')



@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        age = int(request.form['age'])
        gender = request.form['gender'].strip().lower()
        height = float(request.form['height'])
        weight = float(request.form['weight'])
        hypertension = request.form['hypertension'] == 'yes'
        diabetes = request.form['diabetes'] == 'yes'
        meal_type = request.form['meal_type'].strip().lower()
        weight_goal = request.form['weight_goal'].strip().lower()

        # Check if username already exists in the main_user table
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor(dictionary=True)
            cursor.execute('SELECT * FROM main_user WHERE username = %s', (username,))
            existing_user = cursor.fetchone()
            
            if existing_user:
                flash('Username already exists', 'danger')
            else:
                # Insert new user with login and personal details
                cursor.execute('''INSERT INTO main_user (username, password, age, gender, height, weight, hypertension, diabetes, meal_type, weight_goal) 
                                  VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)''', 
                               (username, password, age, gender, height, weight, hypertension, diabetes, meal_type, weight_goal))
                conn.commit()
                flash('Registration successful. Please log in.', 'success')
            cursor.close()
            conn.close()
        
        return redirect(url_for('login'))
    
    return render_template('register.html')




@app.route('/', methods=['GET', 'POST'])
def home():
    # Check if user is logged in first
    if 'user_id' not in session:
        return redirect(url_for('login'))  # If not logged in, redirect to login page

    user_id = session['user_id']  # Retrieve logged-in user ID
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    # Fetch user details from the database (login + personal details)
    cursor.execute('SELECT * FROM main_user WHERE user_id = %s', (user_id,))
    user_details = cursor.fetchone()

    if request.method == 'POST':
        regenerate = request.form.get('regenerate') == 'true'

        if regenerate and user_details:
            # If regenerate, use existing session data
            age = user_details['age']
            gender = user_details['gender']
            height = user_details['height']
            weight = user_details['weight']
            hypertension = user_details['hypertension']
            diabetes = user_details['diabetes']
            meal_type = user_details['meal_type']
            weight_goal = user_details['weight_goal']

        else:
            # If not regenerating, process the form submission
            age = int(request.form['age'])
            gender = request.form['gender'].strip().lower()
            height = float(request.form['height'])
            weight = float(request.form['weight'])
            hypertension = request.form['hypertension'] == 'yes'
            diabetes = request.form['diabetes'] == 'yes'
            meal_type = request.form['meal_type'].strip().lower()
            weight_goal = request.form['weight_goal'].strip().lower()

            # Store user details in session
            session['user_details'] = {
                'age': age,
                'gender': gender,
                'height': height,
                'weight': weight,
                'hypertension': hypertension,
                'diabetes': diabetes,
                'meal_type': meal_type,
                'weight_goal': weight_goal
            }

            # Insert or update user details in the database
            cursor.execute('''UPDATE user_details 
                              SET age=%s, gender=%s, height=%s, weight=%s, hypertension=%s, diabetes=%s, meal_type=%s, weight_goal=%s
                              WHERE user_id=%s''',
                           (age, gender, height, weight, hypertension, diabetes, meal_type, weight_goal, user_id))
            conn.commit()

        cursor.close()
        conn.close()

        # If the form was submitted, show results
        if regenerate or user_details:
            # Generate weekly meal and exercise plans
            daily_calories = calculate_calories(age, height, weight, gender)
            target_calories = adjust_calories_for_goal(daily_calories, weight_goal)

            days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            weekly_meals = {}
            weekly_exercises = {}

            for day in days:
                meals = {
                    "Breakfast": select_item(filter_meal_type_and_conditions(nutrition_data[nutrition_data['Breakfast'] == 1], meal_type, hypertension, diabetes)),
                    "Lunch": select_item(filter_meal_type_and_conditions(nutrition_data[nutrition_data['Lunch'] == 1], meal_type, hypertension, diabetes)),
                    "Dinner": select_item(filter_meal_type_and_conditions(nutrition_data[nutrition_data['Dinner'] == 1], meal_type, hypertension, diabetes)),
                    "Beverage": select_item(filter_meal_type_and_conditions(nutrition_data[nutrition_data['Bevrage'] == 1], meal_type, hypertension, diabetes)),
                    "Snack": select_item(filter_meal_type_and_conditions(nutrition_data[nutrition_data['Snack'] == 1], meal_type, hypertension, diabetes)),
                    "Curry": select_item(filter_meal_type_and_conditions(nutrition_data[nutrition_data['Curry'] == 1], meal_type, hypertension, diabetes))
                }
                weekly_meals[day] = {meal: (item['food_name'] if item is not None else "N/A") for meal, item in meals.items()}
                weekly_exercises[day] = select_personalized_exercise(gender, weight_goal, hypertension, diabetes)

            return render_template('result.html', weekly_meals=weekly_meals, weekly_exercises=weekly_exercises, target_calories=target_calories)

    # If no form was submitted or no details found, return to index.html with user details pre-filled
    return render_template('index.html', user_details=user_details if user_details else {})

@app.route('/video_feed')
def video_feed():
    global streaming
    streaming = True
    exercise = request.args.get('exercise', 'default') 
    return Response(generate_frames(exercise), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/posture_analysis')
def posture_analysis():
    exercise = request.args.get("exercise", "squat")
    return Response(generate_frames(exercise), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/posture')
def posture_page():
    return render_template('posture.html')

@app.route('/posture', methods=['GET', 'POST'])
def analyze_posture():
    if request.method == 'POST':
        exercise_name = request.form.get('exercise_name')
        return redirect(url_for('live_posture', exercise_name=exercise_name))  # Redirect to live feed

    return render_template('posture.html', posture_result=None, live_feed=False)

@app.route('/live_posture')
def live_posture():
    exercise_name = request.args.get('exercise_name', 'Unknown Exercise')
    return render_template('posture.html', posture_result=f"Analyzing posture for {exercise_name}...", live_feed=True)
@app.route('/stop')
def stop_stream():
    global streaming
    streaming = False  # Stop streaming
    return "Stream Stopped"



logging.basicConfig(level=logging.DEBUG)

@app.route('/logout', methods=['POST'])
def logout():
    try:
        method = request.method  # Extract method first
        logging.debug(f"Logout requested with method: {method}")  # Use logging
        
        session.clear()
        return redirect(url_for('login'))
    except Exception as e:
        logging.error(f"Error during logout: {repr(e)}")  # Use logging
        return redirect(url_for('login'))


if __name__ == '__main__':
    app.run(debug=True)
