import os
import pickle
from flask import Flask
import flask
#from flask_bootstrap import Bootstrap
import joblib
import numpy as np
import pandas as pd
from webapp.db import get_db
import tensorflow as tf
from tensorflow import keras
from PIL import Image


# Load pre-trained machine learning model.
base_path = "/Users/nesara/Documents/aim/cs/projects/product-segmentation-ecommerce/webapp/static/";
load_model_path = base_path + "models/model.h5"
#from keras.models import load_model
model = keras.models.load_model(load_model_path)


def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
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
            product_segmented_image_path = None
            return(flask.render_template('base.html', images=all_images, selected_image=selected_image,
                filter_product_category=filter_product_category, product_segmented_image_path=product_segmented_image_path))


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

                # Get segmentation mask from model (Run inference on model)
                full_image_path = base_path + "images/" + selected_image["image_path"]
                product_segmented_image_path = get_model_output_segmentation_mask(full_image_path)
                product_segmented_image_path = "product_segmented_image/" + "test.png"
                # product_segmented_image_path = None
                # <img id="overlay_image" src="{{url_for('static', filename=product_segmented_image_path)}}" />
                
            else:
                selected_image = None
                product_segmented_image_path = None
            
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
                filter_product_category=filter_product_category, product_segmented_image_path=product_segmented_image_path))

    #Bootstrap(app)

    from . import db
    db.init_app(app)

    return app




def get_model_output_segmentation_mask(image_path):
    """
    Run inference, return segmented product
    """
    # Preprocess image
    input_image = preprocess_image(image_path)
    
    # Add extra dimension for model input
    input_image_model = np.expand_dims(input_image, axis=0)
    
    # Run inference
    pred_mask = model.predict(input_image_model)
    
    # Select category with highest activation
    seg_mask = tf.argmax(pred_mask, axis=-1)
    seg_mask = seg_mask[..., tf.newaxis]
    seg_mask = seg_mask[0]

    # Set transparency to background pixels detected in mask
    product_segmented = get_segmented_product(input_image, seg_mask)

    # Save segmented product as image
    product_segmented_image_path = "product_segmented_image/" + "product_segmented_image.png"
    product_segmented.save(base_path + product_segmented_image_path, "PNG")

    return product_segmented_image_path


def preprocess_image(image_path):
    # Load image as array
    raw_image = tf.keras.preprocessing.image.load_img(image_path)
    arr_image = tf.keras.preprocessing.image.img_to_array(raw_image)
    # Resize
    input_image = tf.image.resize(arr_image, (128, 128))
    # Normalise
    input_image = tf.cast(input_image, tf.float32) / 255.0

    # Add extra dimension for model input
    # input_image = np.expand_dims(input_image, axis=0)

    return input_image

def get_segmented_product(input_image, seg_mask):
    img = tf.keras.preprocessing.image.array_to_img(input_image)
    img = img.convert("RGBA")
    imgpixdata = img.load()

    msk = tf.keras.preprocessing.image.array_to_img(seg_mask)
    msk = msk.convert("RGBA")
    mskpixdata = msk.load()

    # Set transparency to background pixels detected in mask
    width, height = img.size
    for y in range(height):
        for x in range(width):
            if mskpixdata[x, y] == (0, 0, 0, 255):
                imgpixdata[x, y] = (0, 0, 0, 0)

    return img
