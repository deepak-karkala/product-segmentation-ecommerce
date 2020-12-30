# Image segmentation using Deep Learning for e-commerce applications
The project aimed at using image segmentation for products on e-commerce websites. It involved creating dataset by scraping product images, using transfer learning to fine-tune the pre-trained deep learning model to perform segmentation of products from images. The Model predictions were served using FLASK Webapp, containerised using Docker and deployed on AWS through CI/CD Pipeline.


## Project Overview
The project consisted of the following steps,

* Data Collection: Scraping product images from ecommerce websites
* Data Labelling: Using Detectron2 to obtain labels (masks) for training custom model
* Data Pre-processing: Image resize, Data augmentation, Normalisation
* Model Fitting and Training:
* Building image segmentation model using MobileNet as base model
* Fine-tuning last layer of pre-trained model
* Tuning hyperparameters
* Model Serving: Using FLASK to deploy and serve Model predictions using REST API
* Container: Using Docker to containerise the Web Application
* Production: Using AWS CI/CD Pipeline for continuous integration and deployment.

![Project Overview](/doc_images/ecom_prod_seg_overview.png)


## Project Resources
<div class="row">
  <div class="col-lg-8 col-md-10 col-sm-10 col-12 mx-auto">
    <ul>
      <li>The <a href="http://ec2-65-0-106-104.ap-south-1.compute.amazonaws.com/">AWS Webapp</a> for this project</li>
      <li>Run this code on <a href="https://colab.research.google.com/drive/1snDybPVSYFC2swpiexTW5_F6Prv_fQ_F?usp=sharing">Google Colab</a></li>
      <li>View Source on <a href="https://github.com/deepak-karkala/product-segmentation-ecommerce">Github</a></li>
      <li>Docker Container for the project: dkarkala01/ecom-prod-seg-app</li>
    </ul>
  </div>
</div>

## Project Objective
<div class="row">
  <div class="col-lg-8 col-md-10 col-sm-10 col-12 mx-auto">
    <p class="p_no_top_gap">
      <b>The Problem: </b>One of the major factor that holds people back while shopping online for furniture and home decor products is that they won't be able to touch, see in person and feel how the product would fit into their home. The user has to go solely by the images of products posted on the ecommerce websites. This makes the user more hesitant while shopping online for such products.
    </p>
    <p class="p_no_top_gap">
      <b>The Solution: </b>The project aims to address this issue by providing an option for the user to segment the product from the image       and then visualise how it fits in their own home. This is done by using an Image Segmentation Model which can separate the predefined set of product                  categories from the rest of the image, which can then be placed on a video stream of the user's room for visualisation. 
    </p>
  </div>
</div>

<div class="row image_row">
  <div class="col-lg-6 col-md-10 col-sm-10 col-12 mx-auto">
    <img class="block_diagram" src="doc_images/ecom_prod_seg_goal.png">
  </div>
</div>


## Project Challenges and constraints
<div class="row">
  <div class="col-lg-8 col-md-10 col-sm-10 col-12 mx-auto">
    <p class="p_no_top_gap">
      <ul>
        <li><b>Limited Data Availability</b>: One of the major challenges that most enterprises face while planning to adopt Machine Learning into their system is the lack of availability of massive datasets necessary to train powerful Models. While in this project, although more data could have been acquired by scraping more product images, this is avoided in order to be constrained to work with a very limited dataset. Will this small dataset be enough to get satisfactory performance is one of the questions to be answered through this project.</li>
        <li><b>Non-resuability of pre-trained models</b>: While there are already many pre-trained Image Segmentation Models which can be used for this purpose, there are two major challenges,
          <ul>
            <li><b>Custom classes</b>: Most pre-trained models would be trained on standard datasets such as MS-COCO, PASCAL-VOC whereas for our application there is a need to segment product categories which are not part of those datasets. As a result, a custom model will need to be built on top of the pre-trained model and the final layers of this model will then have to be trained using custom dataset.</li>
            <li><b>Model evaluation time</b>: The pre-trained models are often massive in size since they have been trained to classify a wide variety of objects. Due to this, such models are slow in nature. However one of the challenges in this application is to have really fast response time such that user can view the segmented product in real-time.</li>
          </ul>
          As a result of this, such pre-trained models cannot be used for this application, instead a custom model which can perform image segmentation on custom product categories with very small size and evaluation time will need to be developed which forms the goal of this project.
        </li>
      </ul>
    </p>
  </div>
</div>


## End Result
<div class="row">
  <div class="col-lg-8 col-md-10 col-sm-10 col-12 mx-auto">
    <p class="p_no_top_gap">
      The end result of this project is shown below. The screen capture (from the app deployed on AWS) shows an example use case of the user selecting a product on a typical ecommerce website, the Image segmentation model returning the segmented product after which the user can then adjust the size and position of the product placed on the video stream of user's room.
    </p>
  </div>
</div>

<div class="row image_row">
  <div class="col-lg-8 col-md-10 col-sm-10 col-12 mx-auto video_row">
    <video controls preload="auto" autoplay="autoplay" loop="loop" muted="muted">
      <source src="images/demo_2x.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
  </div>
</div>