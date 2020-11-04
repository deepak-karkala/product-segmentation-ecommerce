import os
import pickle
from flask import Flask
import flask
#from flask_bootstrap import Bootstrap
import sklearn
import joblib
import pandas as pd
from webapp.db import get_db

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

            # Option 1: Get image paths from db
            db = get_db()
            all_images = db.execute(
                    'SELECT id, category, image_path FROM product'
                ).fetchall()
            # Option 2: Get static image paths 
            #all_images = os.listdir(os.path.join(app.static_folder, "images"))
            selected_image = None
            filter_product_category = None
            return(flask.render_template('base.html', images=all_images, selected_image=selected_image, filter_product_category=filter_product_category))


        # Return prediction output
        if flask.request.method == 'POST':
            db = get_db()

            # Check if a product has been selected
            if "product_radio" in flask.request.form:
                selected_image_id = flask.request.form['product_radio']
                print(selected_image_id)
                # Get selected image from db
                selected_image = db.execute(
                        'SELECT id, category, image_path FROM product p WHERE p.id = ' + selected_image_id 
                    ).fetchone()
                print(selected_image)
            else:
                selected_image = None
            
            if ("filter_product_button" in flask.request.form) and (flask.request.form["filter_product_button"] != "All"):
                filter_product_category = flask.request.form["filter_product_button"]
                # Show all images if not filtered
                print(filter_product_category)
                all_images = db.execute(
                    'SELECT id, category, image_path FROM product p WHERE p.category = "' + filter_product_category + '"'
                ).fetchall()
                print(all_images)

                # If products are filtered by category, set previously selected image to None
                #selected_image = None
            else:
                # Show all images if not filtered
                all_images = db.execute(
                    'SELECT id, category, image_path FROM product'
                ).fetchall()
                filter_product_category = None
            
            print(selected_image)


            #all_images = os.listdir(os.path.join(app.static_folder, "images"))
            #return(flask.render_template('base.html', images=all_images))
            return(flask.render_template('base.html', images=all_images, selected_image=selected_image,
                filter_product_category=filter_product_category))



    #Bootstrap(app)

    from . import db
    db.init_app(app)

    return app