import cv2 as cv
import pandas as pd
import os
from pathlib import Path


# TODO: Maybe don't call this jpg if we don't now if this works with loaded jpgs?

def mat2jpg(mat, scale = 1, quality = 95):
    """
    
    Converts matrix of color values to jpg formatted byte vector.

    Parameters
    ----------
    mat : ndarray
        2D/3D matrix with byte (unit8) values.
    scale : TYPE, optional
        DESCRIPTION. The default is 1.
    quality : TYPE, optional
        DESCRIPTION. The default is 95.

    Returns
    -------
    jpg : ndarray
        Vector with byte values, formatted as jpg.

    """  
    if scale != 1:
        height = int(mat.shape[0] * scale)
        width = int(mat.shape[1] * scale)
        
        mat = cv.resize(mat, (width, height))
    # TODO: Warn if scale > 1 ?
    
    # Quality is jpg image quality, it's between 0 and 100,
    # both values inclusive, higher is better
    # 95 is opencv default
    quality = round(quality)
    if quality not in range(0, 101):
        raise Exception("Quality must be between 0 and 100 (inclusive)!")

    
    
    # This returns a vector of bytes, which seems to be the compressed image
    # and also contains information on its format
    # So far no documentation was found that the quality flag and the value are
    # to be a list, this is by analogy to the C++ example
    succ, jpg = cv.imencode(".jpg", mat, [cv.IMWRITE_JPEG_QUALITY, quality])
    
    if not succ:
        raise Exception("Failed to encode image!")
    
    return jpg

def jpg2mat(jpg):
    """
    Converts jpg formatted data to matrix of color values.

    Parameters
    ----------
    jpg : ndarray
        Vector with byte values, formatted as jpg.

    Returns
    -------
    mat : ndarray
        2D/3D matrix with byte (unit8) values.

    """
    # This apparently doesn't have a success return value,
    # if not succesfull mat is empty
    mat = cv.imdecode(jpg, cv.IMREAD_UNCHANGED)
    
    if mat is None:
        raise Exception("Could not decode image!")
    
    return mat

def extractJpg(files = None, throw = False, name = "cameraImage"):
    """
    Extracts the stored camera image from a measurement file and save it to disc.

    Parameters
    ----------
    files : str,[str], optional
        Path(es) to the datafiles. If None, all files in the current folder
        are processed. The default is None.
    throw : bool, optional
        Should there be an exception if no image is found in a file?
        Otherwise these are silently ignored. The default is False.
    name : str, optional
        The index of the image in the datafile. The default is "CameraImage".

    Returns
    -------
    None.

    """
    if type(files) == str:
        d = pd.read_pickle(files)
        try:
            jpg = d[name]
        except:
            if throw:
                raise Exception("Could not extract image for " + 
                      d["file_timestamp"] + "!")
            
            return
        
        p = Path(files)
        with open(p.with_suffix(".jpg"), "wb") as w:
                w.write(jpg)
    else:
        if files is None:
            files = [f for f in os.listdir(".") if f.endswith(".s")]
        
        for f in files:
            d = pd.read_pickle(f)
            try:
                jpg = d[name]
            except:
                if throw:
                    raise Exception("Could not extract image for " + 
                          d["file_timestamp"] + "!")
                
                continue
        
            p = Path(f)
            
            # So the thing encoded by opencv is apparently really a valid jpg!
            with open(p.with_suffix(".jpg"), "wb") as w:
                w.write(jpg)