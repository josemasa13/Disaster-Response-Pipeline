# Disaster Response Pipeline Project
<img src="https://upload.wikimedia.org/wikipedia/en/a/a6/Attached_to_figure-eight-dot-com.png" width="35%"><img src="https://i2.wp.com/blog.udacity.com/wp-content/uploads/2019/03/480-white.png?fit=1220%2C480&ssl=2" width="65%">

## Project summary
This project is part of the Data Science nanodegree program offered by Udacity in collaboration with Figure Eight(data providers). 

Following a disaster, typically there will be lots of communication between people either direct or by social media right at the time when response organizations have the least capacity to filter and pull out the least important ones. Often it is really one in every thousand messages that might be relevant to the disaster response organizations. 
The components built for this project are the following:<br/><br/>
-ETL script(data/process_data.py) to clean data and load it to a database file.<br/><br/>
-Training classifier script(models/train_classifier.py) which builds a machine learning model using Natural Language Processing techniques to classify messages into several categories and stores this model into a pickle file.<br/><br/>
-Website using flask backend to perform classifying task with any text provided into one/several categories and which provides a description of the training dataset using visualizations.<br/><br/>


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`<br/>
        The table name by default is "Message"
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
        

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
