from io import BytesIO

def get_image_uri(plt):
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_uri = buffer.getvalue()
    buffer.close()
    return image_uri


from flask import Flask, request, render_template, redirect, url_for
import numpy as np
import folium
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import matplotlib.pyplot as plt
import io
import base64
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import json
from textblob import TextBlob
from geopy.exc import GeocoderUnavailable
import os
from openai import OpenAI


app = Flask(__name__)

# Initialize the geocoder
geolocator = Nominatim(user_agent="service_request_app")

# Define parameters for Thane region
thane_center = (19.2183, 72.9781)  # Latitude and longitude of Thane city center
num_workers_per_dept = 20  # Number of workers per department

# Generate synthetic department locations for Thane region
num_departments = 10
dept_locations = np.random.uniform(low=(thane_center[0]-0.2, thane_center[1]-0.2),
                                   high=(thane_center[0]+0.2, thane_center[1]+0.2),
                                   size=(num_departments, 2))

# Define function to calculate distance between two points
def calculate_distance(loc1, loc2):
    return geodesic(loc1, loc2).kilometers

# Function to estimate time to reach destination based on distance
def estimate_time(distance):
    # Assuming average speed of 30 km/h for workers
    time_hours = distance / 15
    time_minutes = int((time_hours - int(time_hours)) * 60)
    if int(time_hours) == 0 and time_minutes == 0:
        return "Less than 1 minute"
    return f"{int(time_hours)} hours {time_minutes} minutes"

# Reverse geocoding function to get the address from coordinates
def get_address_from_coords(coords):
    location = geolocator.reverse(coords)
    return location.address if location else "Unknown"

# Genetic Algorithm for optimizing service request routing
def genetic_algorithm(user_location, dept_locations, num_workers):
    # Find the closest department to the user
    min_dist = float('inf')
    closest_dept_id = None
    for dept_id, dept_loc in enumerate(dept_locations):
        dist = calculate_distance(user_location, dept_loc)
        if dist < min_dist:
            min_dist = dist
            closest_dept_id = dept_id
    
    # Assign the worker from the closest department to the user
    assigned_dept_id = closest_dept_id
    assigned_worker_id = np.random.randint(0, num_workers)
    
    return assigned_dept_id, assigned_worker_id

# Function to preprocess feedback data and train Random Forest Classifier
def train_classifier(feedback_data, labels):
    if len(feedback_data) < 2:
        # Handle the case where the dataset is too small
        raise ValueError("The dataset is too small to split into training and testing sets.")
    else:
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(feedback_data, labels, test_size=0.2, random_state=42)
    
    # Train Random Forest Classifier
    classifier = RandomForestClassifier()
    classifier.fit(X_train, y_train)
    
    return classifier


# Function to predict feedback using trained classifier
def predict_feedback(classifier, feedback_data):
    # Make predictions using the trained classifier
    predictions = classifier.predict(feedback_data)
    
    return predictions

@app.route('/')
def home():
    # Render the HTML form for service requests
    return render_template('service_request_form.html')

@app.route('/service_request', methods=['POST'])
def service_request():
    # Extract service request data from form fields
    area_name = request.form['area_name']
    service_type = request.form['service_type']

    try:
        # Geocode the area name to obtain its latitude and longitude
        location = geolocator.geocode(area_name)
        if location:
            user_location = (location.latitude, location.longitude)
        else:
            raise GeocoderUnavailable("Geocoding service unavailable")
    except GeocoderUnavailable:
        # If geocoding fails, use a default location (e.g., city center)
        user_location = thane_center

    # Use genetic algorithm to find the closest worker
    assigned_dept_id, assigned_worker_id = genetic_algorithm(user_location, dept_locations, num_workers_per_dept)

    # Display worker locations on a map
    m = folium.Map(location=user_location, zoom_start=12)

    # Add markers for workers in the assigned department
    assigned_dept_loc = dept_locations[assigned_dept_id]
    for worker_id in range(num_workers_per_dept):
        if worker_id == assigned_worker_id:
            # Color the assigned worker differently and add a popup message
            icon = folium.Icon(color='red')
            worker_loc = assigned_dept_loc
            distance_to_user = calculate_distance(user_location, worker_loc)
            time_to_reach = estimate_time(distance_to_user)
            worker_address = get_address_from_coords(worker_loc)
            popup_message = f'<div style="padding: 10px;"><b>Worker Location:</b> {worker_address}<br><b>Estimated Time to Reach:</b> {time_to_reach}</div>'
            folium.Marker(worker_loc, popup=popup_message, icon=icon).add_to(m)
        else:
            icon = folium.Icon(color='green')
            worker_loc = assigned_dept_loc + np.random.uniform(low=(-0.1, -0.1), high=(0.1, 0.1))
            folium.Marker(worker_loc, icon=icon).add_to(m)

    # Add marker for the user's location with a popup message containing buttons
    user_popup_message = 'Please provide us your feedback.<br><button onclick="window.location.href=\'/feedback_page\'" style="background-color: #4CAF50; border: none; color: white; padding: 10px 24px; text-align: center; text-decoration: none; display: inline-block; font-size: 16px; margin: 4px 2px; cursor: pointer;">YES</button><button onclick="window.location.href=\'/\'" style="background-color: #f44336; border: none; color: white; padding: 10px 24px; text-align: center; text-decoration: none; display: inline-block; font-size: 16px; margin: 4px 2px; cursor: pointer;">NO</button>'
    folium.Marker(user_location, popup=user_popup_message, icon=folium.Icon(color='blue')).add_to(m)

    # Zoom the map onto the user's location
    m.fit_bounds([user_location], padding=(30, 30))

    # Save the map to a file
    m.save('templates/user_map.html')

    # Redirect the user to the map page
    return redirect(url_for('show_map'))


@app.route('/show_map')
def show_map():
    # Render the map page
    return render_template('user_map.html')

@app.route('/feedback_page')
def feedback_page():
    # Render the feedback question page
    feedback_questions = [
        "Question 1: How satisfied are you with the service provided?",
        "Question 2: Did the worker arrive on time?",
        "Question 3: Was the quality of work satisfactory?",
        "Question 4: How likely are you to recommend our service to others?",
        "Question 5: Overall, how satisfied are you with our service?"
    ]
    return render_template('feedback_questions.html', feedback_questions=feedback_questions)

# Generate random training data
training_data = pd.DataFrame(np.random.randint(1, 6, size=(30000, 4)), columns=['Q1', 'Q2', 'Q3', 'Q4'])
labels = np.random.randint(0, 2, size=30000)  # Random labels (binary classification)

# Train Random Forest Classifier
classifier = RandomForestClassifier()
classifier.fit(training_data, labels)

# Function to make predictions using the trained classifier
def predict_feedback(classifier, data):
    predictions = classifier.predict(data)
    return predictions

# Example usage
test_data = pd.DataFrame(np.random.randint(1, 6, size=(5, 4)), columns=['Q1', 'Q2', 'Q3', 'Q4'])
predictions = predict_feedback(classifier, test_data)
print(predictions)
'''
@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    # Collect feedback responses from the form
    q1 = int(request.form['Q1'])
    q2 = int(request.form['Q2'])
    q3 = int(request.form['Q3'])
    q4 = int(request.form['Q4'])
    #q5_comments = request.form['Q5'].split(';')  # Split comments separated by semicolon
    
    # Preprocess the data if necessary
    feedback_data = pd.DataFrame({
        "Q1": [q1] * 30000,  # Satisfaction level for Question 1
        "Q2": [q2] * 30000,  # Satisfaction level for Question 2
        "Q3": [q3] * 30000,  # Satisfaction level for Question 3
        "Q4": [q4] * 30000,  # Satisfaction level for Question 4
    })

    # Placeholder for labels based on feedback_data
    # You can modify this based on your actual use case
    labels = feedback_data.mean(axis=1).round().astype(int)

    # Train Random Forest Classifier
    classifier = train_classifier(feedback_data, labels)

    # Make predictions using the trained classifier
    predictions = predict_feedback(classifier, feedback_data)


    # Calculate statistics for each question
    # Question 1
    q1_average_rating = feedback_data["Q1"].mean()
    q1_distribution = feedback_data["Q1"].value_counts(normalize=True) * 100

    # Question 2
    q2_yes_percentage = (feedback_data["Q2"] > 1).mean() * 100  # Considering ratings from 1 to 5 as "Yes"
    q2_no_percentage = 100 - q2_yes_percentage

    # Question 3
    q3_yes_percentage = (feedback_data["Q3"] > 1).mean() * 100  # Considering ratings from 1 to 5 as "Yes"
    q3_no_percentage = 100 - q3_yes_percentage


    # Question 4
    q4_average_likelihood_rating = feedback_data["Q4"].mean()
    q4_distribution = feedback_data["Q4"].value_counts(normalize=True) * 100

    # Question 5
    #q5_average_rating = feedback_data["Q5"].mean()

    # Generate Plotly charts
    q1_pie_chart = px.pie(names=q1_distribution.index.map(str), values=q1_distribution.values, title='Question 1: Satisfaction with Service Provided')
    q2_bar_chart = px.bar(x=['Very Satisfied', 'Not Very Satisfied'], y=[q2_yes_percentage, q2_no_percentage], title='Question 2: Worker Arrival Time', labels={'x': 'Response', 'y': 'Percentage'})
    q3_bar_chart = px.bar(x=['Very Satisfied', 'Not Very Satisfied'], y=[q3_yes_percentage, q3_no_percentage], title='Question 3: Quality of Work', labels={'x': 'Response', 'y': 'Percentage'})
    q4_pie_chart = px.pie(names=q4_distribution.index.map(str), values=q4_distribution.values, title='Question 4: Likelihood to Recommend')
    
    # Serialize Plotly charts to JSON
    q1_pie_chart_json = q1_pie_chart.to_json()
    q2_bar_chart_json = q2_bar_chart.to_json()
    q3_bar_chart_json = q3_bar_chart.to_json()
    q4_pie_chart_json = q4_pie_chart.to_json()

     # Check if 'Q5' exists in the form data
    if 'Q5' in request.form:
        q5_comments = request.form['Q5'].split(';')  # Split comments separated by semicolon
    else:
        q5_comments = []
        
    print("Q5 comments:", q5_comments)

    # Sentiment analysis on feedback comments
    sentiment_scores = []
    for comment in q5_comments:
        blob = TextBlob(comment)
        sentiment_scores.append(blob.sentiment.polarity)

    # Calculate sentiment distribution
    positive_sentiment = sum(score > 0 for score in sentiment_scores)
    negative_sentiment = sum(score < 0 for score in sentiment_scores)
    neutral_sentiment = len(sentiment_scores) - positive_sentiment - negative_sentiment

    # Render the thank you message for feedback submission
    return "Thank you for your feedback!"
'''

client = OpenAI(
    api_key = ''
    # Defaults to os.environ.get("OPENAI_API_KEY")
)


def get_completion(prompt):
    query = client.chat.completions.create(
        model="gpt-3.5-turbo",
       messages=[ 
           {"role": "system", "content": "You are a helpful AI assistant."},
           {"role": "user", "content": prompt}],
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
    )
    response = query.choices
    return response

@app.route('/chatbot/', methods=['GET', 'POST'])
def index():
    context = {}
    if request.method == "POST":
        prompt = request.form.get('prompt')  # Changed from request.POST to request.form
        response = get_completion(prompt)
        context['response'] = response

    return render_template('chatbot.html', context=context)

if __name__ == '__main__':
    app.run(debug=True,port=4500)
