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

## Project Setup
### Project Resources
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

### Project Objective
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


### Project Challenges and constraints
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


### End Result
<div class="row">
  <div class="col-lg-8 col-md-10 col-sm-10 col-12 mx-auto">
    <p class="p_no_top_gap">
      The end result of this project is shown below. The screen capture (from the app deployed on AWS) shows an example use case of the user selecting a product on a typical ecommerce website, the Image segmentation model returning the segmented product after which the user can then adjust the size and position of the product placed on the video stream of user's room.
    </p>
  </div>
</div>
<div class="row image_row">
  <div class="col-lg-6 col-md-10 col-sm-10 col-12 mx-auto">
    <img class="block_diagram" src="doc_images/demo_2x.gif">
  </div>
</div>


### Project Considerations
<div class="row">
  <div class="col-lg-8 col-md-10 col-sm-10 col-12 mx-auto">
    <p class="p_no_top_gap">
      Before deciding to use Machine Learning in any application, there are a number of factors to be considered such as what business purpose does the project serve, project constraints, performance constraints, how to evaluate the system performance. The following block diagram describes all the major considerations.
    </p>
  </div>
</div>

<div class="row image_row">
  <div class="col-lg-6 col-md-10 col-sm-10 col-12 mx-auto">
    <img class="block_diagram" src="doc_images/ecom_prod_seg_setup.png">
  </div>
</div>

## Data Considerations and Pipeline

### Data Considerations
<div class="row">
  <div class="col-lg-8 col-md-10 col-sm-10 col-12 mx-auto">
    <p class="p_no_top_gap">
      One of the major challenges in a Machine Learning project is to handle the various parts of the Data Pipeline such as <em>Data Collection</em>, <em>Data Storage</em>, <em> Data Pre-processing and Representation</em>, <em>Data Privacy</em>, <em>Bias in Data</em>. It is important to handle these aspects of Data pipeline and the following block-diagram answers all the questions regarding handling data.
    </p>
  </div>
</div>

<div class="row image_row">
  <div class="col-lg-6 col-md-10 col-sm-10 col-12 mx-auto">
    <img class="block_diagram" src="doc_images/ecom_prod_seg_data.png">
  </div>
</div>

### Data Pipeline
<div class="row">
  <div class="col-lg-8 col-md-10 col-sm-10 col-12 mx-auto">
    <p class="p_no_top_gap">
      After scraping for product images and obtaining reference masks using pre-trained model, the data is stored in a JSON file (paths to images and masks). Pre-processing is then performed on this dataset after which the dataset is divided into 3 sets,
    </p>
    <table class="table table-sm table-bordered">
      <thead>
        <tr>
          <th scope="col">Data</th>
          <th scope="col">Purpose</th>
          <th scope="col">Number of Images</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th scope="row">Training</th>
          <td>To fit Model</td>
          <td>285</td>
        </tr>
        <tr>
          <th scope="row">Validation</th>
          <td>To tune hyperparameters</td>
          <td>95</td>
        </tr>
        <tr>
          <th scope="row">Test</th>
          <td>To evaluate model performance</td>
          <td>95</td>
        </tr>
      </tbody>
    </table>
    <p class="p_no_top_gap">
      As can be observed, this is a very small dataset which is to be used to fine-tune the final layers of a pre-trained Image Segmentation Model.
    </p>
  </div>
</div>
<div class="row image_row">
  <div class="col-lg-4 col-md-10 col-sm-10 col-12 mx-auto">
    <img class="block_diagram" src="doc_images/ecom_prod_seg_datapipeline.png">
  </div>
</div>


## Modeling: Fitting and Training Deep learning models

<div class="row">
  <div class="col-lg-8 col-md-10 col-sm-10 col-12 mx-auto" id="metrics">
    <h3 class="sub-title">Model Evaluation Metrics</h3>
  </div>
</div>


<div class="row">
  <div class="col-lg-8 col-md-10 col-sm-10 col-12 mx-auto">
    <p class="p_no_top_gap">
      The two metrics often used for Image segmentation tasks are 
      <ul>
        <li><b>Pixel wise Classification Accuracy</b>: In this metric, each pixel is regarded as belonging to a class (background or one of the product categories), </li>
        <li><b>Intersection Over Union (IoU)</b>: As described in <a href="https://www.tensorflow.org/api_docs/python/tf/keras/metrics/MeanIoU">Tensorflow documentation</a>, IoU is defined as, IoU = true_positive / (true_positive + false_positive + false_negative).</li>
      </ul>
      In this project, during fitting the model, both the metrics are tracked to see the progress in model training. The second metric, IoU is used to compare model performance on Test dataset.
    </p>
  </div>
</div>

<div class="row">
  <div class="col-lg-8 col-md-10 col-sm-10 col-12 mx-auto" id="baseline">
    <h3 class="sub-title">Baseline Models for Model Comparison</h3>
  </div>
</div>

<div class="row">
  <div class="col-lg-8 col-md-10 col-sm-10 col-12 mx-auto">
    <p class="p_no_top_gap">
      The following are some of the popular Image Segmentation Models. There are two constraints due to which these models cannot be used in this application.
      <ul>
        <li>These models would have been trained on standard datasets and are not capable of performing segmentation on custom product categories used in this application.</li>
        <li>The Model size and evaluation time do not fit the project specifications. In order to be used in a ecommerce website, the model size needs to be small and the evaluation time needs to be very low which would enable an user to view the segmented product instantly.</li>
      </ul>
    </p>
  </div>
</div>

<div class="row">
  <div class="col-lg-8 col-md-10 col-sm-10 col-12 mx-auto">
    <table class="table table-sm table-bordered">
      <thead>
        <tr>
          <th scope="col">Model</th>
          <th scope="col">Size (in MB)</th>
          <th scope="col">Evaluation Time</th>
          <th scope="col">IoU</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th scope="row">Detectron 2</th>
          <td>178 MB</td>
          <td>0.10 seconds</td>
          <td>0.89</td>
        </tr>
        <tr>
          <th scope="row">Mask RCNN</th>
          <td>170 MB</td>
          <td>6.29 seconds</td>
          <td>0.66</td>
        </tr>
      </tbody>
    </table>
  </div>
</div>

<div class="row">
  <div class="col-lg-8 col-md-10 col-sm-10 col-12 mx-auto">
    <p class="p_no_top_gap">
      The goal of this project is to build such a custom model which can serve two purposes sepcifically,
      <ul>
        <li>Able to perform Image segmentation on custom product categories</li>
        <li>The Model should be small in size with very low evaluation time per image.</li>
      </ul>
      The following sections will describe the building, training and evaluation of such a custom model. 
    </p>
  </div>
</div>


<div class="row">
  <div class="col-lg-8 col-md-10 col-sm-10 col-12 mx-auto" id="transfer_learning">
    <h3 class="sub-title">Transfer Learning: Fine-tuning pre-trained model</h3>
  </div>
</div>
<div class="row">
  <div class="col-lg-8 col-md-10 col-sm-10 col-12 mx-auto">
    <p class="p_no_top_gap">
      Although a new custom model is necessary for this application, there is no need for this custom model to be built from scratch. Instead a pre-trained model (trained on massive image datasets such as ImageNet) can be used as a base for the custom model. On top of this base model, additional layers can be added. During training, the weights and bias of the base model are fixed and not changed whereas the final layers which are added will be updated. This process is referred to as <b>Transfer Learning</b> where in only the last few layers are trained. After which, all the layers of the custom model can be trained with a small learning rate. This is referred to as <b>Fine-tuning</b> a pre-trained model. 
    </p>
    <p>The code block to build such a model is shown here,</p>
  </div>
</div>
<div class="row script_row">
  <div class="col-lg-8 col-md-10 col-sm-10 col-12 mx-auto" style="height: 60vh; overflow: scroll;">
    <script src="https://gist.github.com/deepak-karkala/142606e650351778b3f9a09ed817aff2.js"></script>
  </div>
</div>
<div class="row">
  <div class="col-lg-8 col-md-10 col-sm-10 col-12 mx-auto">
    <h3 class="sub-title">Image Segmentation Model: Custom Layers on top of MobileNet Base Model</h3>
  </div>
</div>
<div class="row">
  <div class="col-lg-8 col-md-10 col-sm-10 col-12 mx-auto">
    <p class="p_no_top_gap">
      The Custom Model is built with the following components,
      <ul>
        <li><b>Base Model</b>: MobileNet V2 pre-trained on ImageNet</li>
        <li><b>Custom Layers</b>: Decoder layers of U-Net</li>
      </ul>
      The input image is first passed through downsampling layers of MobileNet and then upsampled (with skip connections) through the decoder layer stack. All the layers of this Custom Model can be seen in the following figure presented here.
    </p>
  </div>
</div>
<div class="row image_row">
  <div class="col-lg-8 col-md-10 col-sm-10 col-12 mx-auto">
    <img src="doc_images/model.png">
  </div>
</div>