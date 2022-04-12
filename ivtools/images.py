import cv2 as cv

# TODO: Maybe don't call this jpg if we don't now if this works with loaded jpgs?

def mat2jpg(mat, scale = 1, quality = 95):
      
    
    if scale != 1:
        height = int(mat.shape[0] * scale)
        width = int(mat.shape[1] * scale)
        
        mat = cv.resize(mat, (width, height))
    # TODO: Warn if scale > 1 ?
    
    # TODO: Quality is jpg image quality, is between 0 and 100, inclusive, higher is better
    
    
    # This returns a vector of bytes, which seems to be the compressed image
    # and also contains information on its format
    # So far no documentation was found that the quality flag and the value are
    # to be a list, this is by analogy to the C++ example
    succ, jpg = cv.imencode(".jpg", mat, [cv.IMWRITE_JPEG_QUALITY, 50])
    
    return jpg

def jpg2mat(jpg):
    # This apparently doesn't have a success return value
    # No idea what the -1 does
    mat = cv.imdecode(jpg, -1)
    
    return mat