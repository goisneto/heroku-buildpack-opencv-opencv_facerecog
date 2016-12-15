/**************************************************************************
*   Copyright (C) 2007 by Robin Hewitt                                    *
*   Copyright (C) 2010 by elsamuko                                        *
*   elsamuko@web.de                                                       *
*                                                                         *
*   This program is free software; you can redistribute it and/or modify  *
*   it under the terms of the GNU General Public License as published by  *
*   the Free Software Foundation; either version 3 of the License, or     *
*   (at your option) any later version.                                   *
*                                                                         *
*   This program is distributed in the hope that it will be useful,       *
*   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
*   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
*   GNU General Public License for more details.                          *
*                                                                         *
*   You should have received a copy of the GNU General Public License     *
*   along with this program; if not, write to the                         *
*   Free Software Foundation, Inc.,                                       *
*   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
***************************************************************************/

// small program to detect and recognize faces in an input image, based on the example of the OpenCV wiki:
// http://opencv.willowgarage.com/wiki/FaceDetection
// and the face recognition example of Robin Hewitt:
// http://www.cognotics.com/opencv/servo_2007_series/index.html

// compile with:
// g++ -O2 -Wall -funroll-loops -lboost_filesystem `pkg-config --cflags opencv` `pkg-config --libs opencv` -o opencv-facerecog opencv-facerecog.cpp

// OpenCV
#include "cv.h"
#include "cvaux.h"
#include "highgui.h"

// C++
#include <iostream>
#include <fstream>
#include <string>
#include <utility>
#include <vector>
#include <map>

// BOOST
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/tokenizer.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem.hpp>

// defines
#define SCALE         1
#define FACE_SIZE    100

#define WORKING_DIR  ".opencv-facerecog"
#define FACES_DIR    "faces"
#define DEBUG_DIR    "debug"

#define CASCADE_FILE "haarcascade_frontalface_alt.xml"
#define FACEDB_FILE  "facedata.xml"
#define NAME_FILE   "names.txt"
#define LOG_FILE     "log.txt"

using namespace std;
using namespace boost;
// simple container class
class Eigenface {
public:
    int nTrainFaces;                // the number of training images
    int nEigens;                    // the number of eigenvalues
    IplImage* pAvgTrainImg;         // the average image
    vector<IplImage*> eigenVectVec; // eigenvectors
    CvMat* eigenValMat;             // eigenvalues
    CvMat* projectedTrainFaceMat;   // projected training faces
};

// global variables
bool DEBUG;                   // debug
posix_time::ptime clock_gl;   // timer
string logFile;               // logfile
string debugDir;              // folder for debug output

// function prototypes
CvSeq*                    detectFaces( IplImage* img, CvMemStorage* storage, CvHaarClassifierCascade* cascade );
vector<IplImage*>         getFaces( IplImage* img, CvMemStorage* storage, CvHaarClassifierCascade* cascade );
void                      saveFaces( const string& imgFile, vector<IplImage*> faces );
void                      addFace( const string& imgFile, vector<IplImage*> faces, const filesystem::path& name );
IplImage*                 getSubimage( const IplImage* img, const CvRect* r );
IplImage*                 resizeImage( const IplImage* image, const double ratio );

bool                      learn( Eigenface& data, const vector<IplImage*>& faceImgVec );
int                       recognize( Eigenface& data, IplImage* image );
void                      doPCA( Eigenface& data, const vector<IplImage*>& faceImgVec );
int                       findNearestNeighbor( const Eigenface& data, float* projectedTestFace );
void                      storeTrainingData( const Eigenface& data, const string& filename );
int                       loadTrainingData( Eigenface& data, const string& filename );
int                       loadFaceImgVector( vector<IplImage*>& faces, const string& dir );

vector<string>            listDir( const filesystem::path& dir );
vector< vector <string> > listSubdir( const filesystem::path& dir );
void                      writeNameFile( const string& filename, const vector< vector <string> >& subFiles );
vector< string >          loadNameFile( const string& filename );
int                       writeLog( const int line, const string& text );
int                       writeMatrix( CvMat* mat, const string& filename );
vector<string>            getNameList( const string& dir );
string                    getFolder( const string& filename );
void                      usage();

// modified C functions from Robin Hewitt
bool learn( Eigenface& data, const vector<IplImage*>& faceImgVec ) {
    // enough faces?
    if( data.nTrainFaces < 2 ) {
        writeLog( __LINE__, "Need 2 or more training faces." );
        writeLog( __LINE__, "Input file contains only " + lexical_cast<string>( data.nTrainFaces ) );
        return false;
    }

    // do PCA on the training faces
    doPCA( data, faceImgVec );

    // project the training images onto the PCA subspace
    data.projectedTrainFaceMat = cvCreateMat( data.nTrainFaces, data.nEigens, CV_32FC1 );
    int offset = data.projectedTrainFaceMat->step / sizeof( float );

    for( int i = 0; i < data.nTrainFaces; ++i ) {
        cvEigenDecomposite( faceImgVec[i],                                      // Input object
                            data.nEigens,                                        // Number of eigen objects.
                            & ( *( data.eigenVectVec.begin() ) ),                // Array of IplImage input objects
                            0, 0,                                                // I/O flags, userData
                            data.pAvgTrainImg,                                   // Averaged object
                            //projectedTrainFaceMat->data.fl + i*nEigens);
                            data.projectedTrainFaceMat->data.fl + i * offset );  // Calculated coefficients
    }

    // check for double faces
    if( DEBUG ) {
        for( int i = 0; i < data.nEigens; ++i ) {
            writeLog( __LINE__, "*(data.projectedTrainFaceMat->data.fl + " + lexical_cast<string>( i )
                      + "*offset) = " + lexical_cast<string>( *( data.projectedTrainFaceMat->data.fl + i * offset ) ) );
        }
    }

    map<float, int> projVals;

    for( int i = 0; i < data.nEigens; ++i ) {
        projVals[*( data.projectedTrainFaceMat->data.fl + i * offset )]++;
    }

    for( map<float, int>::iterator i = projVals.begin(); i != projVals.end(); ++i ) {
        if( i->second > 1 ) {
            writeLog( __LINE__, "Error: Database contains probably one face two times: " + lexical_cast<string>( i->first ) );
            return false;
        }
    }

    return true;
}

void doPCA( Eigenface& data, const vector<IplImage*>& faceImgVec ) {
    // set the number of eigenvalues to use
    data.nEigens = data.nTrainFaces - 1;

    // allocate the eigenvector images
    CvSize faceImgSize;
    faceImgSize.width  = faceImgVec[0]->width;
    faceImgSize.height = faceImgVec[0]->height;
    data.eigenVectVec = vector<IplImage*>( data.nEigens );

    for( int i = 0; i < data.nEigens; ++i ) {
        data.eigenVectVec[i] = cvCreateImage( faceImgSize, IPL_DEPTH_32F, 1 );
    }

    data.eigenValMat = cvCreateMat( 1, data.nEigens, CV_32FC1 );
    data.pAvgTrainImg = cvCreateImage( faceImgSize, IPL_DEPTH_32F, 1 );
    CvTermCriteria calcLimit = cvTermCriteria( CV_TERMCRIT_ITER, data.nEigens, 1 );

    // compute average image, eigenvalues, and eigenvectors
    cvCalcEigenObjects( data.nTrainFaces,                       // Number of source objects.
                        ( void* ) & ( *( faceImgVec.begin() ) ), // Array of IplImage input objects
                        ( void* ) & ( *( data.eigenVectVec.begin() ) ), // Array of eigen objects
                        CV_EIGOBJ_NO_CALLBACK,                  // I/O flags
                        0, 0,                                   // I/O buffer, userData
                        &calcLimit,                             // Criteria that determine when to stop calculation of eigen objects
                        data.pAvgTrainImg,                      // Averaged object.
                        data.eigenValMat->data.fl );            // Pointer to the eigenvalues array

    cvNormalize( data.eigenValMat, data.eigenValMat, 1, 0, CV_L1, 0 );
}

int recognize( Eigenface& data, IplImage* image ) {
    writeLog( __LINE__, "data.pAvgTrainImg->depth  = " +
              lexical_cast<string>( ( data.pAvgTrainImg )->depth ) );
    writeLog( __LINE__, "image->depth  = " +
              lexical_cast<string>( image->depth ) );
    writeLog( __LINE__, "image->width  = " +
              lexical_cast<string>( image->width ) );
    writeLog( __LINE__, "image->height = " +
              lexical_cast<string>( image->height ) );

    // project the image onto the PCA subspace
    float* projectedTestFace = ( float * )cvAlloc( data.nEigens * sizeof( float ) );
    cvEigenDecomposite( image,                          // Input object
                        data.nEigens,                    // Number of eigen objects.
                        & ( *( data.eigenVectVec.begin() ) ), // Array of IplImage input objects
                        0, 0,                            // I/O flags, userData
                        data.pAvgTrainImg,               // Averaged object
                        projectedTestFace );             // Calculated coefficients

    // check for errors
    if( DEBUG ) {
        for( int i = 0; i < data.nEigens; ++i ) {
            writeLog( __LINE__, "projectedTestFace[" + lexical_cast<string>( i ) + "] = " + lexical_cast<string>( projectedTestFace[i] ) );
        };
    }

    for( int i = 0; i < data.nEigens; ++i ) {
        if( isnan( projectedTestFace[i] ) ) {
            writeLog( __LINE__, "Error: Database contains probably one face two times" );
            return -1;
        }
    }

    // find best hit
    int iNearest = findNearestNeighbor( data, projectedTestFace );
    cvFree( &projectedTestFace );
    writeLog( __LINE__, "iNearest = " + lexical_cast<string>( iNearest ) );
    return iNearest;
}

int loadFaceImgVector( vector<IplImage*>& faceImgVec, const string& dir ) {
    int nFaces = 0;
    vector< vector <string> > subFiles = listSubdir( dir );

    // count faces
    for( vector< vector <string> >::const_iterator i = subFiles.begin(); i != subFiles.end(); ++i ) {
        nFaces += i->size();

        if( DEBUG ) writeLog( __LINE__, "nFace: " + lexical_cast<string>( i->size() ) );
    }

    writeLog( __LINE__, "nFaces: " + lexical_cast<string>( nFaces ) );

    if( nFaces == 0 ) return 0;

    faceImgVec = vector<IplImage*>( nFaces );

    // store the face images in an array
    int count = 0;

    for( vector< vector <string> >::const_iterator i = subFiles.begin(); i != subFiles.end(); ++i ) {
        for( vector <string>::const_iterator j = i->begin(); j != i->end(); ++j ) {
            unsigned int number  = i - subFiles.begin();
            faceImgVec[count] = cvLoadImage( ( char* )( *j ).c_str(), CV_LOAD_IMAGE_GRAYSCALE );

            if( !faceImgVec[count] ) {
                writeLog( __LINE__, "Can't load image from " + ( *j ) );
                return 0;
            }

            if( DEBUG ) writeLog( __LINE__, lexical_cast<string>( number ) + " " + *j );

            count++;
        }
    }

    return nFaces;
}

int findNearestNeighbor( const Eigenface& data, float * projectedTestFace ) {
    double leastDistSq = 999999999; //DBL_MAX;
    int iNearest = 0;

    for( int iTrain = 0; iTrain < data.nTrainFaces; iTrain++ ) {
        double distSq = 0;

        for( int i = 0; i < data.nEigens; ++i ) {
            float d_i = projectedTestFace[i] - data.projectedTrainFaceMat->data.fl[iTrain * data.nEigens + i];

            if( DEBUG ) writeLog( __LINE__, "d_i: " + lexical_cast<string>( d_i ) );

            distSq += d_i * d_i / data.eigenValMat->data.fl[i]; // Mahalanobis
            // distSq += d_i*d_i; // Euclidean
        }

        if( DEBUG ) writeLog( __LINE__, "distSq: " + lexical_cast<string>( distSq ) );

        if( distSq < leastDistSq ) {
            leastDistSq = distSq;
            iNearest = iTrain;
        }
    }

    writeLog( __LINE__, "leastDistSq: " + lexical_cast<string>( leastDistSq ) );
    return iNearest;
}

int loadTrainingData( Eigenface& data, const string& filename ) {
    // create a file-storage interface
    CvFileStorage* fileStorage = cvOpenFileStorage( filename.c_str(), 0, CV_STORAGE_READ );

    if( !fileStorage ) {
        writeLog( __LINE__, "Can't open " + filename );
        return 0;
    }

    writeLog( __LINE__, filename + " loaded" );

    // load data
    data.nEigens               =          cvReadIntByName( fileStorage, 0, "nEigens",               0 );
    data.nTrainFaces           =          cvReadIntByName( fileStorage, 0, "nTrainFaces",           0 );
    data.eigenValMat           = ( CvMat* )    cvReadByName( fileStorage, 0, "eigenValMat",           0 );
    data.projectedTrainFaceMat = ( CvMat* )    cvReadByName( fileStorage, 0, "projectedTrainFaceMat", 0 );
    data.pAvgTrainImg          = ( IplImage* ) cvReadByName( fileStorage, 0, "avgTrainImg",           0 );
    writeLog( __LINE__, "pAvgTrainImg->width  = " +
              lexical_cast<string>( data.pAvgTrainImg->width ) );
    writeLog( __LINE__, "pAvgTrainImg->height = " +
              lexical_cast<string>( data.pAvgTrainImg->height ) );

    data.eigenVectVec = vector<IplImage*>( data.nTrainFaces );

    for( int i = 0; i < data.nEigens; ++i ) {
        string varname = "eigenVect_" + lexical_cast<string>( i );
        data.eigenVectVec[i] = ( IplImage* ) cvReadByName( fileStorage, 0, varname.c_str(), 0 );
    }

    // release the file-storage interface
    cvReleaseFileStorage( &fileStorage );
    return 1;
}

void storeTrainingData( const Eigenface& data, const string& filename ) {
    // create a file-storage interface
    CvFileStorage * fileStorage = cvOpenFileStorage( filename.c_str(), 0, CV_STORAGE_WRITE );

    // store all the data
    cvWriteInt( fileStorage, "nEigens",               data.nEigens );
    cvWriteInt( fileStorage, "nTrainFaces",           data.nTrainFaces );
    cvWrite( fileStorage, "eigenValMat",           data.eigenValMat,           cvAttrList( 0, 0 ) );
    cvWrite( fileStorage, "projectedTrainFaceMat", data.projectedTrainFaceMat, cvAttrList( 0, 0 ) );
    cvWrite( fileStorage, "avgTrainImg",           data.pAvgTrainImg,          cvAttrList( 0, 0 ) );

    for( int i = 0; i < data.nEigens; ++i ) {
        string varname = "eigenVect_" + lexical_cast<string>( i );
        cvWrite( fileStorage, varname.c_str(), data.eigenVectVec[i], cvAttrList( 0, 0 ) );
    }

    if( DEBUG ) {
        writeLog( __LINE__, "nEigens = " + lexical_cast<string>( data.nEigens ) );
        writeLog( __LINE__, "nTrainFaces = " + lexical_cast<string>( data.nTrainFaces ) );
        // cvSave(string(debugDir + "/" + "eigenValMat.cvmat").c_str(), data.eigenValMat);
        // cvSave(string(debugDir + "/" + "projectedTrainFaceMat.cvmat").c_str(), data.projectedTrainFaceMat);
        writeMatrix( data.eigenValMat, debugDir + "/" + "eigenValMat.txt" );
        writeMatrix( data.projectedTrainFaceMat, debugDir + "/" + "projectedTrainFaceMat.txt" );
        cvSaveImage( string( debugDir + "/" + "avgTrainImg.pgm" ).c_str(), data.pAvgTrainImg );

        for( int i = 0; i < data.nEigens; ++i ) {
            string varname = "eigenVect_" + lexical_cast<string>( i );
            cvSaveImage( string( debugDir + "/" + varname + ".pgm" ).c_str(), data.eigenVectVec[i] );
        }
    }

    // release the file-storage interface
    cvReleaseFileStorage( &fileStorage );
}

// program itself
int main( int argc, char** argv ) {

    // setup env vars
    string workingDir  = filesystem::current_path().string() + "/" + WORKING_DIR;
    filesystem::create_directory( workingDir );
    string facesDir    = workingDir  + "/" + FACES_DIR;
    string cascadeFile = workingDir  + "/" + CASCADE_FILE;
    string facesdbFile = workingDir  + "/" + FACEDB_FILE;
    string nameFile    = workingDir  + "/" + NAME_FILE;
    logFile            = workingDir  + "/" + LOG_FILE;
    debugDir           = workingDir  + "/" + DEBUG_DIR;
    filesystem::create_directory( debugDir );

    // setup logger
    posix_time::ptime tmp( posix_time::microsec_clock::universal_time() );
    clock_gl = tmp;
    writeLog( __LINE__, "" );
    writeLog( __LINE__, "Starting ..." );

    if( argc > 1 ) {

        Eigenface data;               // feature container
        vector<IplImage*> faceImgVec; // vector of face images
        IplImage* image = 0;          // image for det & rec

        string imageIn;
        string name;
        bool add_face = false;
        bool detect_only = false;
        bool build_only = false;
        bool resize = false;
        DEBUG = false;

        // create face listings
        vector< vector<string> > subFiles = listSubdir( facesDir );
        vector<string> nameList = getNameList( facesDir );

        // set options
        for( int i = 1; i < argc; ++i ) {
            const char *sw = argv[i];

            if( !strcmp( sw, "-h" ) || !strcmp( argv[i], "--help" ) ) {
                usage();
                writeLog( __LINE__, "Ending ..." );
                return 0;

            } else if( !strcmp( sw, "-D" ) || !strcmp( sw, "--debug" ) ) {
                DEBUG = true;

            } else if( !strcmp( sw, "-d" ) || !strcmp( sw, "--detect-only" ) ) {
                detect_only = true;

            } else if( !strcmp( sw, "-b" ) || !strcmp( sw, "--build-db" ) ) {
                build_only = true;

            } else if( !strcmp( sw, "-r" ) || !strcmp( sw, "--resize" ) ) {
                resize = true;

            } else if( !strcmp( sw, "-c" ) || !strcmp( sw, "--cascade" ) ) {
                if( i + 1 >= argc ) {
                    writeLog( __LINE__, "No cascade file" );
                    usage();
                    writeLog( __LINE__, "Ending ..." );
                    return 1;
                }

                cascadeFile = argv[++i];

            } else if( !strcmp( sw, "-a" ) || !strcmp( sw, "--add" ) ) {
                if( i + 1 >= argc ) {
                    writeLog( __LINE__, "No name given" );
                    usage();
                    writeLog( __LINE__, "Ending ..." );
                    return 1;
                }

                add_face = true;
                name = argv[++i];

            } else {
                imageIn += sw;
            }
        }

        if( DEBUG ) {
            writeLog( __LINE__, "subFiles:" );

            for( vector< vector<string> >::iterator i = subFiles.begin(); i != subFiles.end(); ++i ) {
                for( vector<string>::iterator j = i->begin(); j != i->end(); ++j ) {
                    writeLog( __LINE__, *j );
                }
            }

            writeLog( __LINE__, "nameList:" );

            for( vector<string>::iterator i = nameList.begin(); i != nameList.end(); ++i ) {
                writeLog( __LINE__, *i );
            }
        }

        // build db
        if( build_only ) {
            writeNameFile( nameFile, subFiles );

            // load
            data.nTrainFaces = loadFaceImgVector( faceImgVec, facesDir );

            if( data.nTrainFaces > 0 ) {
                // learn
                if( learn( data, faceImgVec ) ) {
                    // store the recognition data as an xml file
                    storeTrainingData( data, facesdbFile );
                } else {
                    cout << "Error: Could not build faces database" << endl;
                }
            } else {
                writeLog( __LINE__, "No faces in " + facesDir );
            }
            
            for( vector<IplImage*>::iterator i = faceImgVec.begin(); i != faceImgVec.end(); ++i ) {
                cvReleaseImage( &(*i) );
            }

            // detect and/or recognize
        } else {
            // some checks
            if( imageIn.size() == 0 ) {
                writeLog( __LINE__, "Error: Not enough arguments" );
                usage();
                writeLog( __LINE__, "Ending ..." );
                return 0;
            }

            writeLog( __LINE__, "imageIn:  " + imageIn );
            writeLog( __LINE__, "cascade:  " + cascadeFile );

            // load the HaarClassifierCascade
            CvHaarClassifierCascade* cascade = ( CvHaarClassifierCascade* )cvLoad( cascadeFile.c_str(), 0, 0, 0 );

            if( !cascade ) {
                writeLog( __LINE__, "Error: Could not load classifier cascade" );
                writeLog( __LINE__, "Ending ..." );
                return -1;
            }

            // load image
            image = cvLoadImage( imageIn.c_str(), 1 );

            if( !image ) {
                writeLog( __LINE__, "Error: Could not load image" );
                writeLog( __LINE__, "Ending ..." );
                return -1;
            }

            // optionally resize image
            if( resize ) {
                double ratio = max( image->height, image->width ) / 1024.0;

                if( ratio > 1.0 ) image = resizeImage( image, ratio );
            }

            // detect faces
            CvMemStorage* storage = cvCreateMemStorage( 0 );
            vector<IplImage*> faceImgVec = getFaces( image, storage, cascade );
            cvReleaseMemStorage( &storage );
            cvReleaseHaarClassifierCascade( &cascade );
            
            if( faceImgVec.empty() ) {
                writeLog( __LINE__, "No faces found" );
                writeLog( __LINE__, "Ending ..." );
                return 0;
            }

            if( add_face ) {
                // add face to faces dir
                addFace( imageIn, faceImgVec, facesDir + "/" + name );

            } else if( detect_only ) {
                // save faces as pgm files
                saveFaces( imageIn, faceImgVec );

            } else {
                // load xml with faces db
                if( !loadTrainingData( data, facesdbFile ) ) {
                    writeLog( __LINE__, "Error: Could not load training data" );
                    return -1;
                }

                vector<string> names = loadNameFile( nameFile );

                if( ( int )names.size() != data.nTrainFaces ) {
                    writeLog( __LINE__, "Error: " + lexical_cast<string>( names.size() ) + " entries in " + nameFile );
                    writeLog( __LINE__, "Error: " + lexical_cast<string>( data.nTrainFaces ) + " entries expected" );
                    return -1;
                }

                // iterate over detected faces and try to recognize them
                writeLog( __LINE__, "Starting recognition" );

                for( vector<IplImage*>::iterator i = faceImgVec.begin(); i != faceImgVec.end(); ++i ) {
                    writeLog( __LINE__, "(*i)->depth  = " + lexical_cast<string>( ( *i )->depth ) );
                    int pos = recognize( data, *i );

                    if( pos != -1 ) {
                        writeLog( __LINE__, lexical_cast<string>( pos ) );
                        writeLog( __LINE__, names[pos] );
                        // resolve face id to name
                        cout << names[pos] << endl;
                    } else {
                        // not yet used
                        writeLog( __LINE__, "Face not recognized" );
                    }
                }
            }
            
            for( vector<IplImage*>::iterator i = faceImgVec.begin(); i != faceImgVec.end(); ++i ) {
                cvReleaseImage( &(*i) );
            }

            cvReleaseImage( &image );
        }
    } else {
        usage();
    }

    writeLog( __LINE__, "Ending ..." );
    return 0;
}


// additional helper functions
// small logger
int writeLog( const int line, const string& text ) {
    posix_time::ptime tmp( posix_time::microsec_clock::universal_time() );
    ofstream out( logFile.c_str(), ios::out | ios::app );

    if( out.is_open() ) {
        out << to_simple_string( tmp - clock_gl ) << "\t" << "L" << line << ": " << text << endl;
        out.close();
    } else {
        cout << "L" << __LINE__ << ": " << "Unable to open " << logFile << endl;
    }

    return 0;
}

// debug function to write out the OpenCV matrix as Octave matrix
int writeMatrix( CvMat* mat, const string& filename ) {
    bool error = false;
    ofstream out( filename.c_str() );   // Open file for writing

    if( out.is_open() ) {
        // out.setf( ios::scientific );
        // out.precision( 6 );
        out << "# Created by opencv-facerecog\n";
        out << "# name:  " << filesystem::basename( filename ) << "\n";
        out << "# type:  matrix\n";
        out << "# rows: " << mat->rows << "\n";
        out << "# columns: " << mat->cols << "\n";

        for( int i = 0; i < mat->rows; ++i ) {
            for( int j = 0; j < mat->cols; ++j ) {
                out << cvmGet( mat, i, j );
            }

            out << "\n";
        }

        out.close();
    } else {
        error = true;
        writeLog( __LINE__, "Unable to open " + filename );
    }

    writeLog( __LINE__, filename + " written" );

    return error;  //zero, if there is no problem
}

// returns relative folder name
string getFolder( const string& filename ) {
    string ret;
    tokenizer<char_separator<char> > tok( filename, char_separator<char>( "/" ) );
    tokenizer<char_separator<char> >::iterator tok_iter;

    for( tok_iter = tok.begin(); boost::next( tok_iter ) != tok.end(); ++tok_iter ) {
        ret = *tok_iter;
    }

    return ret;
}

// detects all faces in an image
CvSeq* detectFaces( IplImage* img, CvMemStorage* storage, CvHaarClassifierCascade* cascade ) {
    cvClearMemStorage( storage );
    CvSeq* faces;

    if( cascade ) {
        faces =  cvHaarDetectObjects( img,                      // Image to detect objects in
                                      cascade,                  // Haar classifier cascade in internal representation
                                      storage,                  // Memory storage to store the resultant sequence of the object candidate rectangles
                                      1.1,                      // The factor by which the search window is scaled between the subsequent scans, 1.1 means increasing window by 10 %
                                      3,                        // Minimum number (minus 1) of neighbor rectangles that makes up an object
                                      CV_HAAR_DO_CANNY_PRUNING, // Mode of operation
                                      cvSize( 40, 40 ) );       // Minimum window size
        writeLog( __LINE__, "Found " + lexical_cast<string>( faces->total ) + " face(s)" );
        return faces;
    }

    return 0;
}

// saves detected faces as filename.crop_#.pgm
void saveFaces( const string& imgFile, vector<IplImage*> faces ) {
    for( vector<IplImage*>::iterator i = faces.begin(); i != faces.end(); ++i ) {
        string filename = filesystem::basename( imgFile ) + "_crop_" +
                          lexical_cast<string>( ( i - faces.begin() ) ) + ".pgm";
        writeLog( __LINE__, "Writing " + filename );
        cvSaveImage( filename.c_str(), *i );
    }
}

// add first detected face to faces dir
void addFace( const string& imgFile, vector<IplImage*> faces, const filesystem::path& name ) {
    string filename = filesystem::basename( imgFile ) + "_crop_0.pgm";
    writeLog( __LINE__, "Writing " + name.string() + "/" + filename );
    filesystem::create_directory( name );
    cvSaveImage( ( name.string() + "/" + filename ).c_str(), *( faces.begin() ) );
}

// returns all detected faces as images
vector<IplImage*> getFaces( IplImage* img, CvMemStorage* storage, CvHaarClassifierCascade* cascade ) {
    CvSeq* faces = detectFaces( img, storage, cascade );
    vector<IplImage*> faceArray  = vector<IplImage*>( faces->total );

    for( vector<IplImage*>::iterator i = faceArray.begin(); i != faceArray.end(); ++i ) {
        CvRect* r = ( CvRect* )cvGetSeqElem( faces, ( i - faceArray.begin() ) );
        *i = getSubimage( img, r );
    }

    return faceArray;
}

// returns a b/w image with dim FACE_SIZExFACE_SIZE
IplImage* getSubimage( const IplImage* img, const CvRect* r ) {
    IplImage* tmp = cvCreateImage( cvSize( img->width, img->height ), IPL_DEPTH_8U, 1 );
    cvCvtColor( img, tmp, CV_BGR2GRAY );

    IplImage* crop = cvCreateImage( cvSize( FACE_SIZE, FACE_SIZE ), IPL_DEPTH_8U, 1 );

    cvSetImageROI( tmp, *r );
    cvResize( tmp, crop, CV_INTER_CUBIC );

    cvResetImageROI( tmp );
    cvReleaseImage( &tmp );

    writeLog( __LINE__, "crop->depth  = " + lexical_cast<string>( crop->depth ) );
    return crop;
}

IplImage* resizeImage( const IplImage* image, const double ratio ) {
    IplImage* imageR = cvCreateImage( cvSize( ( int )image->width / ratio, ( int )image->height / ratio ), image->depth, image->nChannels );
    writeLog( __LINE__, "Resize image from w=" + lexical_cast<string>( image->width ) + ", h=" + lexical_cast<string>( image->height ) );
    writeLog( __LINE__, "               to w=" + lexical_cast<string>( imageR->width ) + ", h=" + lexical_cast<string>( imageR->height ) );
    cvResize( image, imageR, CV_INTER_CUBIC );
    return imageR;
}

// returns an array of the dirs content
vector<string> listDir( const filesystem::path& dir ) {
    vector<string> files;

    if( filesystem::exists( dir ) ) {
        filesystem::directory_iterator end;

        for( filesystem::directory_iterator iter( dir ); iter != end; iter++ ) {
            files.push_back( iter->path().string() );
        }
    }

    return files;
}

// returns an array of the dirs content
vector< vector <string> > listSubdir( const filesystem::path& dir ) {
    vector< vector <string> > subFiles;
    vector<string> files = listDir( dir );

    for( vector<string>::iterator i = files.begin(); i != files.end(); ++i ) {
        subFiles.push_back( listDir( *i ) );
    }

    return subFiles;
}

// returns name look-up list
vector<string> getNameList( const string& dir ) {
    vector<string> list;
    vector<string> subDirs = listDir( dir );
    vector<string> tmp;

    for( vector<string>::iterator i = subDirs.begin(); i != subDirs.end(); ++i ) {
        tmp = listDir( *i );

        for( vector<string>::iterator j = tmp.begin(); j != tmp.end(); ++j ) {
            list.push_back( *j );
        }
    }

    return list;
}

// writes file with names
void writeNameFile( const string& filename, const vector< vector <string> >& subFiles ) {
    if( filename.size() != 0 ) {
        ofstream out( filename.c_str() );   // Open file for writing

        if( out.is_open() ) {
            for( vector< vector <string> >::const_iterator i = subFiles.begin(); i != subFiles.end(); ++i ) {
                for( vector <string>::const_iterator j = i->begin(); j != i->end(); ++j ) {
                    out << getFolder( *j ) << "\n";
                }
            }

            out.close();
        } else {
            writeLog( __LINE__, "Unable to open " + filename );
        }

        writeLog( __LINE__, filename + " written" );
    } else {
        writeLog( __LINE__, "Size of filename is zero" );
    }
}

// loads file with names
vector< string > loadNameFile( const string& filename ) {
    vector< string > retVector;

    if( filename.size() == 0 ) {
        writeLog( __LINE__, "Size of filename is zero" );
    } else {
        ifstream in( filename.c_str() );
        string line;

        if( !in ) {
            writeLog( __LINE__, "Unable to open " + filename );
        } else {
            while( getline( in, line ) ) {
                retVector.push_back( line );
            }

            writeLog( __LINE__, filename + " read" );
        }
    }

    return retVector;
}

// default output
void usage() {
    cout << "Usage: opencv-facerecog [options] imageIn" << endl;
    cout << "Options:" << endl;
    cout << "       -h           or  --help              This message" << endl;
    cout << "       -D           or  --debug             Debug output" << endl;
    cout << "       -d           or  --detect-only       Detect faces, writes them out as filename_crop_#.pgm" << endl;
    cout << "       -a NAME      or  --add NAME          Adds first detected face to faces folder" << endl;
    cout << "       -b           or  --build-db          Build face database" << endl;
    cout << "       -c FILE.XML  or  --cascade FILE.XML  Cascade file, default: ~/" << WORKING_DIR << "/" << CASCADE_FILE << endl;
    cout << "       -r           or  --resize            Use smaller image for faster detection" << endl;
    cout << endl;
}
