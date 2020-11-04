import os
import pickle
from flask import Flask
import flask
#from flask_bootstrap import Bootstrap
import sklearn
import joblib
import pandas as pd

# Load pre-trained machine learning model.
base_path = "/Users/nesara/Documents/aim/cs/projects/airbnb-data-science/webapp_alternate_search/";
#with open(base_path + 'static/model/decision_tree.pkl', 'rb') as f:
#    model = pickle.load(f)
#model = joblib.load(base_path + "static/model/fullpipeline_linearregression.pkl")


def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite'),
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    # Landing page
    @app.route('/', methods=['GET', 'POST'])
    def hello():
        # return 'Hello, World!'
        
        # Return landing page
        if flask.request.method == 'GET':
            images = os.listdir(os.path.join(app.static_folder, "images"))
            return(flask.render_template('base.html', images=images))


        # Return prediction output
        if flask.request.method == 'POST':
            selected_image = flask.request.form['product_radio']
            images = os.listdir(os.path.join(app.static_folder, "images"))
            return(flask.render_template('base.html', images=images, selected_image=selected_image))


            """
            # Read variables from form
            temperature = flask.request.form['temperature']
            humidity = flask.request.form['humidity']
            windspeed = flask.request.form['windspeed']

            # Create input to Model from form data
            input_variables = pd.DataFrame([[temperature, humidity, windspeed]],
                                           columns=['temperature', 'humidity', 'windspeed'],
                                           dtype=float)

            # Inference: Get prediction from Model
            prediction = model.predict(input_variables)[0]

            # Return Model output
            return flask.render_template('main.html',
                                         original_input={'Temperature':temperature,
                                                         'Humidity':humidity,
                                                         'Windspeed':windspeed},
                                         result=prediction,
                                         )
            """

    #Bootstrap(app)

    from . import db
    db.init_app(app)

    return app