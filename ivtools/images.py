import cv2 as cv

# TODO: Maybe don't call this jpg if we don't now if this works with loaded jpgs?

def mat2jpg(mat, scale = 1, quality = 95):
      
    
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
    # This apparently doesn't have a success return value,
    # if not succesfull mat is empty
    mat = cv.imdecode(jpg, cv.IMREAD_UNCHANGED)
    
    if mat is None:
        raise Exception("Could not decode image!")
    
    return mat