#!/bin/sh

WORKING_DIR=`echo "$HOME/.opencv-facerecog"`
FACES_DIR=`echo "$WORKING_DIR/faces"`

# create dirs
if [ ! -d $WORKING_DIR ]; then
    mkdir $WORKING_DIR
    echo "$WORKING_DIR created"
fi

if [ ! -d $FACES_DIR ]; then
    mkdir $FACES_DIR
    echo "$FACES_DIR created"
fi

echo "cp haarcascade_frontalface_alt.xml $WORKING_DIR"
cp haarcascade_frontalface_alt.xml $WORKING_DIR

# compile
make clean
make

if [ -f opencv-facerecog ]; then
    echo
    echo "Detect Waldo's, Mona's and Wright's face and copy it into the faces dir"
    echo
    ./opencv-facerecog -D -a Waldo Waldo.jpg
    ./opencv-facerecog -D -a Mona Mona.jpg
    ./opencv-facerecog -D -a Wright Wright.jpg
    
    echo "Rebuild faces db"
    ./opencv-facerecog -D -b
    
    echo
    echo "Recognize faces in Mona.jpg:"
    ./opencv-facerecog -D Mona.jpg
    echo
    echo "Recognize faces in Wright.jpg:"
    ./opencv-facerecog -D Wright.jpg
    echo
    echo "Recognize faces in Waldo.jpg:"
    ./opencv-facerecog -D Waldo.jpg
    echo
    
    echo "See $WORKING_DIR/log.txt for details"
    sleep 1
fi

# tidy up
echo
echo "Tidy up"
make tidy


