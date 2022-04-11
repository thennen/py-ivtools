import cv2 as cv

class MikrOkular:
    # (width, height)
    resolutions = {"low": (1280, 720),
                   "mid": (1280, 1024),
                   "high": (1920, 1080)}
    
    # (min, max) both extreme values are valid
    ranges = {"brightness": (-127, 128),
              "contrast": (0, 30),
              "hue": (-180, 180),
              "saturation": (0, 127),
              "gamma": (20, 250),
              "sharpness": (0, 60),
              "exposure:": (-7, -4)}
    
    codes = {"brightness": cv.CAP_PROP_BRIGHTNESS,
              "contrast": cv.CAP_PROP_CONTRAST,
              "hue": cv.CAP_PROP_HUE,
              "saturation": cv.CAP_PROP_SATURATION,
              "gamma": cv.CAP_PROP_GAMMA,
              "sharpness": cv.CAP_PROP_SHARPNESS,
              "exposure:": cv.CAP_PROP_EXPOSURE}
    
    # count initializations ? 
    
    ###
    def __init__(self):
        # ID?
        self.camera = cv.VideoCapture(0, cv.CAP_DSHOW)
        if not self.camera.isOpened():
            # This fails if the CamLabLite software is open
            # Or camera is not plugged in
            raise Exception("Could not connect to MikrOkular camera!")
    
    def __del__(self):
        self.camera.release()
    
    ###
    def showImg(self):
        ret, frame = self.camera.read()
        cv.imshow("MikrOkular", frame)
        cv.waitKey()
        
    def liveView(self):
        def makePropTrack(name):
            cv.createTrackbar(name, "Controls", 50, 100, lambda x: self.setProperty(name, x/100))
        
        # Good?
        def makeResTrack():
            res = {0: "low", 1: "mid", 2: "high"}
            cv.createTrackbar("Resolution", "Controls", 0, 2, lambda x: self.setResolution(res[x]))
        
        # Needs separate windows, otherwise looks weird
        cv.namedWindow("Video")
        cv.namedWindow("Controls")
        #cv.namedWindow("Video", cv.WINDOW_NORMAL | cv.WINDOW_FREERATIO | cv.WINDOW_GUI_EXPANDED)
        
        # This resets all properties, do we want that?
        makePropTrack("Brightness")
        makePropTrack("Contrast")
        makePropTrack("Hue")
        makePropTrack("Saturation")
        makePropTrack("Gamma")
        makePropTrack("Sharpness")
      
        makeResTrack()
      
        while True:
            succ, frame = self.camera.read()
            cv.imshow("Video", frame)
            #cv.resizeWindow("Video", 400, 400)
            if cv.waitKey(1) == ord("q"):
                break
            
        cv.destroyAllWindows()
    
    ###
    def setResolution(self, res = "high"):
        # check here
        
        width = self.resolutions[res.casefold()][0]
        height = self.resolutions[res.casefold()][1]
        
        succ = self.camera.set(cv.CAP_PROP_FRAME_WIDTH, width)
        succ = self.camera.set(cv.CAP_PROP_FRAME_HEIGHT, height)
        
    # Sould this give pixels or string?
    def getResolution(self):
        pass
    
    ###
    def setPropertyNative(self, prop, value):
        # Check property
        prop = prop.casefold()
        if not prop in self.codes.keys():
            raise Exception("No such property!")
        
        # Check value
        if not type(value) is int:
            raise Exception("Value must be a (signed) integer!")
        
        # Is there a less ugly way then the +1 ?
        if not value in range(self.ranges[prop][0], self.ranges[prop][1]+1):
            raise Exception("Value out of range!")
        
        succ = self.camera.set(self.codes[prop], value)
        if not succ:
            raise Exception("Could not set property!")
        
    def getPropertyNative(self, prop):
        # Check property
        prop = prop.casefold()
        if not prop in self.codes.keys():
            raise Exception("No such property!")
        
        value = self.camera.get(self.codes[prop])
        
        # Properties can only have signed interger values
        return int(value)
         
    def setProperty(self, prop, value):
        # Check value here to give clearer error?
        prop = prop.casefold()
        value = self.ranges[prop][0] + (self.ranges[prop][1]-self.ranges[prop][0]) * value
        value = round(value)
        self.setPropertyNative(prop, value)
    
    def getProperty(self, prop):
        value = self.getPropertyNative(prop)
        value = (value - self.ranges[prop][0])/(self.ranges[prop][1]-self.ranges[prop][0])
        
        return value

    ###
    def close(self):
        self.camera.release()

    # properties truncated to integer, successfully!
    # TODO: check exposure!