# SIGN LANGUAGE DETECTION MODEL
## Demo


https://user-images.githubusercontent.com/89011801/155462438-dc02ded4-3e8d-4f23-8631-8594163aef7f.mp4








**AIM AND OBJECTIVES**

**Aim**

To create a Sign Language detection model which will detect the position
of human hands and then convey the message on the viewfinder of camera
in real time what the particular hand position means.

**Objectives**

  - The main objective of the project is to create a program which can
    be either run on Jetson nano or any pc with YOLOv5 installed and
    start detecting using the camera module on the device.

  - Using appropriate datasets for recognizing and interpreting data
    using machine learning.

  - To show on the optical viewfinder of the camera module what a
    particular position of hand means with respect to Sign Language.

**ABSTRACT**

  - A hand position is classified with respect to Sign Language on the
    basis of where the hand is placed on the body and then shown on the
    viewfinder of the camera what it means.

  - We have completed this project on jetson nano which is a very small
    computational device.

  - A lot of research is being conducted in the field of Computer Vision
    and Machine Learning (ML), where machines are trained to identify
    various objects from one another. Machine Learning provides various
    techniques through which various objects can be detected.

<!-- end list -->

  - > One such technique is to use YOLOv5 with Roboflow model, which
    > generates a small size trained model and makes ML integration
    > easier.

  - > Sign languages are an extremely important communication tool for
    > many deaf and hard-of-hearing people. Sign languages are the
    > native languages of the Deaf community and provide full access to
    > communication.

  - > Sign Language detection model can be of great help to people who
    > are beginners in learning sign language or to those like an
    > elderly who have lost hearing recently.

**INTRODUCTION**

  - > This project is based on Sign Language detection model. We are
    > going to implement this project with Machine Learning and this
    > project can be even run on jetson nano which we have done.

  - > This project can also be used to gather information about what
    > Sign Language a particular person is conveying through his or her
    > hands.

  - > Hand position can be classified into many other names based on the
    > type of language being used based on the image annotation we give
    > in roboflow.

  - > Sign Language detection becomes difficult sometimes on account of
    > people of various age groups, sizes, gender etc doing Sign
    > Language hand position which is harder for model to detect.
    > However, training in Roboflow has allowed us to crop images and
    > change the contrast of certain images to match the time of day for
    > better recognition by the model.

  - > Neural networks and machine learning have been used for these
    > tasks and have obtained good results.

  - > Machine learning algorithms have proven to be very useful in
    > pattern recognition and classification, and hence can be used for
    > Sign Language detection as well.

**LITERATURE REVIEW**

  - > Sign languages (also known as signed languages) are languages that
    > use the visual-manual modality to convey meaning. Sign languages
    > are expressed through manual articulations in combination with
    > non-manual elements. Sign languages are full-fledged natural
    > languages with their own grammar and lexicon.

  - > Wherever communities of deaf people exist, sign languages have
    > developed as useful means of communication, and they form the core
    > of local Deaf cultures. Although signing is used primarily by the
    > deaf and hard of hearing, it is also used by hearing individuals,
    > such as those unable to physically speak, those who have trouble
    > with spoken language due to a disability or condition
    > (augmentative and alternative communication), or those with deaf
    > family members, such as children of deaf adults.

  - > As a sign language develops, it sometimes borrows elements from
    > spoken languages, just as all languages borrow from other
    > languages that they are in contact with. Sign languages vary in
    > how much they borrow from spoken languages.

  - > Although sign languages have emerged naturally in deaf communities
    > alongside or among spoken languages, they are unrelated to spoken
    > languages and have different grammatical structures at their core.

  - > Some experts argue early man likely used signs to communicate long
    > before spoken language was created.

**JETSON NANO COMPATIBILITY**

  - > The power of modern AI is now available for makers, learners, and
    > embedded developers everywhere.

  - > NVIDIA® Jetson Nano™ Developer Kit is a small, powerful computer
    > that lets you run multiple neural networks in parallel for
    > applications like image classification, object detection,
    > segmentation, and speech processing. All in an easy-to-use
    > platform that runs in as little as 5 watts.

  - > Hence due to ease of process as well as reduced cost of
    > implementation we have used Jetson nano for model detection and
    > training.

  - > NVIDIA JetPack SDK is the most comprehensive solution for building
    > end-to-end accelerated AI applications. All Jetson modules and
    > developer kits are supported by JetPack SDK.

  - > In our model we have used JetPack version 4.6 which is the latest
    > production release and supports all Jetson modules.
# Jetson Nano 2GB

![IMG_20220125_115719](https://user-images.githubusercontent.com/89011801/155461874-81d01ca0-11ae-4ac5-9947-a1572f0cc264.jpg)

**PROPOSED SYSTEM**

1.  > Study basics of machine learning and image recognition.

2.  > Start with implementation

<!-- end list -->

  - > Front-end development

  - > Back-end development

<!-- end list -->

3.  > Testing, analyzing and improvising the model. An application using
    > python and Roboflow and its machine learning libraries will be
    > using machine learning to identify which position of hand means
    > what according to Sign Language.

4.  > Use data sets to interpret the hand position and convey what the
    > meaning of hand position is on the viewfinder.

**METHODOLOGY**

The Sign Language detection model is a program that focuses on
implementing real time Sign Language detection.

It is a prototype of a new product that comprises of the main module:

Hand position detection and then showing on viewfinder what the hand
position means according to data fed.

Sign Language Detection Module

This Module is divided into two parts:

1.  > Hand Detection

<!-- end list -->

  - > Ability to detect the location of a person’s hand in any input
    > image or frame. The output is the bounding box coordinates on the
    > detected hand of a person.

  - > For this task, initially the Data set library Kaggle was
    > considered. But integrating it was a complex task so then we just
    > downloaded the images from gettyimages.ae, shutterstock.com and
    > google images and made our own data set.

  - > This Data set identifies person’s hand in a Bitmap graphic object
    > and returns the bounding box image with annotation of name
    > present.

<!-- end list -->

2.  > Position of Hand Detection

<!-- end list -->

  - > Recognition of the hand and what the particular position means.

  - > Hence YOLOv5 which is a model library from roboflow for image
    > classification and vision was used.

  - > There are other models as well but YOLOv5 is smaller and generally
    > easier to use in production. Given it is natively implemented in
    > PyTorch (rather than Darknet), modifying the architecture and
    > exporting and deployment to many environments is straightforward.

  - > YOLOv5 was used to train and test our model for what a particular
    > hand position means. We trained it for 149 epochs and achieved an
    > accuracy of approximately 92%.

**INSTALLATION**

> sudo apt-get remove --purge libreoffice\*
> 
> sudo apt-get remove --purge thunderbird\*
> 
> sudo fallocate -l 10.0G /swapfile1
> 
> sudo chmod 600 /swapfile1
> 
> sudo mkswap /swapfile1
> 
> sudo vim /etc/fstab
> 
> \#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#add line\#\#\#\#\#\#\#\#\#\#\#
> 
> /swapfile1 swap defaults 0 0
> 
> vim \~/.bashrc
> 
> \#\#\#\#\#\#\#\#\#\#\#\#\#add line \#\#\#\#\#\#\#\#\#\#\#\#\#
> 
> export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
> 
> export
> LD\_LIBRARY\_PATh=/usr/local/cuda/lib64${LD\_LIBRARY\_PATH:+:${LD\_LIBRARY\_PATH}}
> 
> export LD\_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
> 
> sudo apt-get update
> 
> sudo apt-get upgrade
> 
> \#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#pip-21.3.1 setuptools-59.6.0
> wheel-0.37.1\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#
> 
> sudo apt install curl
> 
> curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
> 
> sudo python3 get-pip.py
> 
> sudo apt-get install libopenblas-base libopenmpi-dev
> 
> sudo apt-get install python3-dev build-essential autoconf libtool
> pkg-config python-opengl python-pil python-pyrex
> python-pyside.qtopengl idle-python2.7 qt4-dev-tools qt4-designer
> libqtgui4 libqtcore4 libqt4-xml libqt4-test libqt4-script
> libqt4-network libqt4-dbus python-qt4 python-qt4-gl libgle3 python-dev
> libssl-dev libpq-dev python-dev libxml2-dev libxslt1-dev libldap2-dev
> libsasl2-dev libffi-dev libfreetype6-dev python3-dev
> 
> vim \~/.bashrc
> 
> \#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\# add line
> \#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#
> 
> export OPENBLAS\_CORETYPE=ARMV8
> 
> source \~/.bashrc
> 
> sudo pip3 install pillow
> 
> curl -LO
> https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl
> 
> mv p57jwntv436lfrd78inwl7iml6p13fzh.whl
> torch-1.8.0-cp36-cp36m-linux\_aarch64.whl
> 
> sudo pip3 install torch-1.8.0-cp36-cp36m-linux\_aarch64.whl
> 
> sudo python3 -c "import torch; print(torch.cuda.is\_available())"
> 
> git clone --branch v0.9.1 https://github.com/pytorch/vision
> torchvision
> 
> cd torchvision/
> 
> sudo python3 setup.py install
> 
> cd
> 
> git clone https://github.com/ultralytics/yolov5.git
> 
> cd yolov5/
> 
> sudo pip3 install numpy==1.19.4
> 
> history
> 
> \#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#comment torch,PyYAML and
> torchvision in
> requirement.txt\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#
> 
> sudo pip3 install --ignore-installed PyYAML\>=5.3.1
> 
> sudo pip3 install -r requirements.txt
> 
> sudo python3 detect.py
> 
> sudo python3 detect.py --weights yolov5s.pt --source 0
> 
> \#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#Tensorflow\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#
> 
> sudo apt-get install python3.6-dev libmysqlclient-dev
> 
> sudo apt install -y python3-pip libjpeg-dev libcanberra-gtk-module
> libcanberra-gtk3-module
> 
> pip3 install tqdm cython pycocotools
> 
> \#\#\#\#\#\#\#\#\#\#\#\#\#
> https://developer.download.nvidia.com/compute/redist/jp/v46/tensorflow/tensorflow-2.5.0%2Bnv21.8-cp36-cp36m-linux\_aarch64.whl
> \#\#\#\#\#\#
> 
> sudo apt-get install libhdf5-serial-dev hdf5-tools libhdf5-dev
> zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran
> 
> sudo apt-get install python3-pip
> 
> sudo pip3 install -U pip testresources setuptools==49.6.0
> 
> sudo pip3 install -U --no-deps numpy==1.19.4 future==0.18.2
> mock==3.0.5 keras\_preprocessing==1.1.2 keras\_applications==1.0.8
> gast==0.4.0 protobuf pybind11 cython pkgconfig
> 
> sudo env H5PY\_SETUP\_REQUIRES=0 pip3 install -U h5py==3.1.0
> 
> sudo pip3 install -U cython
> 
> sudo apt install python3-h5py
> 
> sudo pip3 install \#install downloaded tensorflow(sudo pip3 install
> --pre --extra-index-url
> https://developer.download.nvidia.com/compute/redist/jp/v46
> tensorflow)
> 
> python3
> 
> import tensorflow as tf
> 
> tf.config.list\_physical\_devices("GPU")
> 
> print(tf.reduce\_sum(tf.random.normal(\[1000,1000\])))
> 
> \#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#mediapipe\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#
> 
> git clone https://github.com/PINTO0309/mediapipe-bin
> 
> ls
> 
> cd mediapipe-bin/
> 
> ls
> 
> ./v0.8.5/numpy119x/mediapipe-0.8.5\_cuda102-cp36-cp36m-linux\_aarch64\_numpy119x\_jetsonnano\_L4T32.5.1\_download.sh
> 
> ls

sudo pip3 install mediapipe-0.8.5\_cuda102-cp36-none-linux\_aarch64.whl

**ADVANTAGES**

  - > As many as 90 percent of deaf children are born to hearing
    > parents, which can make learning sign language a family affair.
    > Hence our Sign Language Detection model can be a great help in
    > beginning to learn it.

  - > Sign Language detection system shows what the position of hands in
    > viewfinder of camera module means with good accuracy.

  - > It can then be used to help people who are just beginning to learn
    > Sign Language or those who don’t know sign language but have a
    > close one who is deaf.

  - > Some children with Autism Spectrum Disorder (ASD) struggle
    > developing verbal communication. Hence people around such children
    > can use Sign Language detection model to understand what the child
    > is saying.

  - > Sign languages can be a great way to gossip without anyone else
    > knowing, and passing on confidential information. Our model here
    > can be used here by training it to just use hand position based on
    > what the individuals have decided what the hand position means.

**APPLICATION**

  - > Detects a person’s hand and then checks what each hand position
    > means in each image frame or viewfinder using a camera module.

  - > Can be used by people who wants to understand deaf people and also
    > used in places like hospitals where the staff is not trained in
    > Sign Language.

  - > Can be used as a reference for other ai models based on Helmet
    > Detection.

**FUTURE SCOPE**

  - > As we know technology is marching towards automation, so this
    > project is one of the step towards automation.

  - > Thus, for more accurate results it needs to be trained for more
    > images, and for a greater number of epochs.

  - > Sign Language detection model can be very easily implemented in
    > smart phones in the form of apps and thus make it possible for
    > everyone to understand deaf and dumb people.

  - > Sign Language detection model can be further improved to show a
    > certain hand position on screen by typing a word or saying it out
    > loud towards the smart phone.

**CONCLUSION**

  - > In this project our model is trying to detect a person’s hand
    > position and then showing it on viewfinder, live as to what the
    > hand position means as we have specified in Roboflow.

  - > This model tries to solve the problem of people who are dumb and
    > deaf and who wants to convey what they are saying to others using
    > Sign Language but can’t as others haven’t learned Sign Language.

  - > The model is efficient and highly accurate and hence works without
    > any lag and also if the data is downloaded can be made to work
    > offline.

**REFERENCE**

1.  > Roboflow:- <https://roboflow.com/>

2.  > Datasets or images used :-
    > https://www.shutterstock.com/search/sign+language

3.  > Google images

**ARTICLES**

1.  > https://www.healthyhearing.com/report/52606-Why-you-should-learn-sign-language-in-the-new-year

2.  > https://blog.ai-media.tv/blog/7-reasons-sign-language-is-awesome

3.  > https://en.wikipedia.org/wiki/Sign\_language
