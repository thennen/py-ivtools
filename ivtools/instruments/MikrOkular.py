import cv2 as cv

class MikrOkular:
    """
    Handles the connection to the Bresser MikrOkular camera.
    """
    
    # Camera has three possible resolution settings, can give other values,
    # but these are mapped to the closest allowed setting
    # (width, height)
    resolutions = {"low": (1280, 720),
                   "mid": (1280, 1024),
                   "high": (1920, 1080)}
    
    # Allowed values for the camera settings
    # (min, max), both extreme values are valid
    ranges = {"brightness": (-127, 128),
              "contrast": (0, 30),
              "hue": (-180, 180),
              "saturation": (0, 127),
              "gamma": (20, 250),
              "sharpness": (0, 60),
              "exposure": (-7, -4)}
    
    codes = {"brightness": cv.CAP_PROP_BRIGHTNESS,
              "contrast": cv.CAP_PROP_CONTRAST,
              "hue": cv.CAP_PROP_HUE,
              "saturation": cv.CAP_PROP_SATURATION,
              "gamma": cv.CAP_PROP_GAMMA,
              "sharpness": cv.CAP_PROP_SHARPNESS,
              "exposure": cv.CAP_PROP_EXPOSURE}
    
    # Opening the camera multiple times is a problem, this leads to an error
    # somewhere in opencv. Issue is subtle, it seems fine to make multiple
    # camera objects at the same time, but making one, deleting it and then
    # making a new one again fails when one tries to take an image
    # This leads to Spyder freezing
    # When the connection is released properly, by using close() before
    # deleting object, everything is fine
    
    # track open connections to contain freezing issues
    openedCams = {}
    
    ###
    def __init__(self, camId = 0, settings = None, res = "high"):
        """
        Connects to the Bresser MikrOkular camera.

        Parameters
        ----------
        camId : int, optional
            If multiple cameras are connected to the PC, one can be selected
            by a number. The default is 0. Multiple cameras are untested.
        settings : dict, optional
            Initial configuration of the camera setting (e.g. brightness).
            If None, they remain whatever the camera is set to. The default is None.
        res : str, optional
            Initial resolution of the camera image.
            Can be
            "high" - (1920, 1080) px,
            "mid" - (1280, 1024) px or
            "low: - (1280, 720) px
            The default is "high".

        Returns
        -------
        None.

        """
        
        self.camId = camId
        
        # Check if a connection to this camera is already open, this happens
        # if it wasn't closed properly. Then return that instead of attempting
        # to open a new one.
        if camId in self.openedCams:
            self.camera =  MikrOkular.openedCams[camId]
        else:
            self.camera = cv.VideoCapture(camId, cv.CAP_DSHOW)
            # There seem to be different backends/drivers available
            # for connecting to the camera, the default one does weird stuff.
            
        if not self.camera.isOpened():
            # This fails if the CamLabLite software is open
            # or camera is not plugged in
            raise Exception("Could not connect to MikrOkular camera!")

        MikrOkular.openedCams.update({camId: self.camera})
        
        if not settings == None:
            self.setAllProperties(settings)
        
        # TODO: For consistency, this should have a default like settings
        self.setResolution(res)
    
    # This doesn't do what one thinks a destructor does, which is also
    # what it should do
    #def __del__(self):
    #    self.camera.release()
    
    def close(self):
        """
        Closes the connection to the camera.

        Returns
        -------
        None.

        """      
        self.camera.release()
        del MikrOkular.openedCams[self.camId]
    
    ##########################################################
    ### Functionality for getting images out of the camera ###
    ##########################################################
   
    def showImg(self, scale = 0.3, gray = False):
        """
        Gets the current image from the camera and shows it in a window.
        
        To close the window, hit any key.
        The image is by default scaled down, so it doesn't take up the whole
        screen.

        Parameters
        ----------
        scale : float, optional
            Image is scaled down by this factor. This should be between
            0 and 1. Larger values untested. The default is 0.3.
        gray : bool, optional
            Image is in grayscale if true. The default is False.

        Returns
        -------
        None.

        """
        frame = self.getImg(gray)
        
        # Scale down image for display purposes, it's basically screen
        # size otherwise
        dispHeight = int(frame.shape[0] * scale)
        dispWidth = int(frame.shape[1] * scale)
        frame = cv.resize(frame, (dispWidth, dispHeight))
        
        cv.imshow("MikrOkular", frame)
        cv.waitKey()
        cv.destroyAllWindows()
        
    def liveView(self, scale = 0.3, gray = False):
        """
        Shows a live video from the camera.
        
        Close by hitting 'q'.
        Video is scaled down by default, so it doesn't take up the whole
        screen.
        Settings can be changed, these are maintained after closing the
        live view.

        Parameters
        ----------
        scale : float, optional
            Video is scaled down by this factor. This should be between
            0 and 1. Larger values untested. The default is 0.3.
        gray : bool, optional
            Image is in grayscale if true. The default is False.

        Returns
        -------
        None.

        """
        # Somehow this is needed here, though it is also in the set/get
        # methods
        if not self.camera.isOpened():
            raise Exception("Connection to camera was closed!")
        
        def makePropTrack(name):
            init = int(100*self.getProperty(name))
            cv.createTrackbar(name, "Controls", init, 100, lambda x: self.setProperty(name, x/100))
            
        def makeExpTrack():
            init = self.getPropertyNative("Exposure") + 7
            cv.createTrackbar("Exposure", "Controls", init, 3, lambda x: self.setPropertyNative("Exposure", x-7))
        
        def makeResTrack():
            res = {0: "low", 1: "mid", 2: "high"}
            init = self.getResolution()
            
            # TODO: use a less screwed up way to do this
            init = [k for k, v in self.resolutions.items() if v==init][0]
            init = int([k for k, v in res.items() if v==init][0])
            cv.createTrackbar("Resolution", "Controls", init, 2, lambda x: self.setResolution(res[x]))
        
        # Needs separate windows, otherwise looks weird
        cv.namedWindow("Video")
        cv.namedWindow("Controls")
        
        makePropTrack("Brightness")
        makePropTrack("Contrast")
        makePropTrack("Hue")
        makePropTrack("Saturation")
        makePropTrack("Gamma")
        makePropTrack("Sharpness")
      
        makeExpTrack()
      
        makeResTrack()
      
        while True:
            frame = self.getImg(gray)
        
            # Scale down image for display purposes, it's basically screen
            # size otherwise
            # Because the live view allows changing the resolution, we must
            # recalculate this
            dispHeight = int(frame.shape[0] * scale)
            dispWidth = int(frame.shape[1] * scale)
            frame = cv.resize(frame, (dispWidth, dispHeight))
            
            cv.imshow("Video", frame)
            
            if cv.waitKey(1) == ord("q"):
                break
            
        cv.destroyAllWindows()
    
    def getImg(self, gray = False):
        """
        Returns the current camera image as a numpy array.

        Parameters
        ----------
        gray : bool, optional
            Image is in grayscale if true. The default is False.

        Returns
        -------
        frame : ndarray
            Matrix with the current pixel values of the camera. 3 dimensional
            if color mode, otherwise 2 dimensional. Color channels are
            (blue, green, red).

        """
        if not self.camera.isOpened():
            raise Exception("Connection to camera was closed!")
        
        # Seems like camera has one frame in buffer that is not current image
        # this should clear it
        succ = self.camera.read()
        
        # Colors are (blue, green, red) apparently
        succ, frame = self.camera.read()

        if not succ:
            raise Exception("Could not acquire image!")
        
        if gray:
            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        return frame

    def saveImg(self, path, gray = False):
        """
        Writes the current camera image to disc.

        Parameters
        ----------
        path : str
            File to write image to. Ending determines file type.
        gray : bool, optional
            Image is in grayscale if true. The default is False.

        Returns
        -------
        None.

        """
        frame = self.getImg(gray)
        
        cv.imwrite(path, frame)
    
    #################################
    ### Configure camera settings ###
    #################################
    
    # TODO: does having that default make sense?
    def setResolution(self, res = "high"):
        """
        Sets the camera resolution.

        Parameters
        ----------
        res : str, optional
            Can be
            "high" - (1920, 1080) px,
            "mid" - (1280, 1024) px or
            "low: - (1280, 720) px
            The default is "high".

        Returns
        -------
        None.

        """
        if not self.camera.isOpened():
            raise Exception("Connection to camera was closed!")
        
        if not res in self.resolutions.keys():
            raise Exception("Resolution must be 'low', 'mid' or 'high'!")
        
        width = self.resolutions[res.casefold()][0]
        height = self.resolutions[res.casefold()][1]
        
        succ = self.camera.set(cv.CAP_PROP_FRAME_WIDTH, width)
        succ = self.camera.set(cv.CAP_PROP_FRAME_HEIGHT, height)
        
        if not succ:
            raise Exception("Could not set resolution!")
        
    def getResolution(self):
        """
        Returns the current camera resolution in pixels.

        Returns
        -------
        (width, height) : int
            Current resolution in pixels.

        """
        if not self.camera.isOpened():
            raise Exception("Connection to camera was closed!")
        width = self.camera.get(cv.CAP_PROP_FRAME_WIDTH)
        height = self.camera.get(cv.CAP_PROP_FRAME_HEIGHT)
    
        return (int(width), int(height))
    
    def setPropertyNative(self, prop, value):
        """
        Sets one camera setting, using the cameras units.

        Parameters
        ----------
        prop : str
            Setting to change.
            Possible values are: 
              "brightness", "contrast", "hue", "saturation", "gamma",
              "sharpness", "exposure"
        value : int
            The value to be set. Possible values are in the cameras own unit,
            and somewhat strange.

        Returns
        -------
        None.

        """
        # Check if property exists
        prop = prop.casefold()
        if not prop in self.codes.keys():
            raise Exception("No such property!")
        
        # Check if value is in range and int
        if not type(value) is int:
            raise Exception("Value must be a (signed) integer!")
        
        # Is there a less ugly way then the +1 ?
        if not value in range(self.ranges[prop][0], self.ranges[prop][1]+1):
            raise Exception("Value out of range!")
        
        if not self.camera.isOpened():
            raise Exception("Connection to camera was closed!")
        
        succ = self.camera.set(self.codes[prop], value)
        if not succ:
            raise Exception("Could not set property!")
        
    def getPropertyNative(self, prop):
        """
        Returns one camera setting, in the cameras units.

        Parameters
        ----------
        prop : str
            Setting to return.
            Possible values are: 
              "brightness", "contrast", "hue", "saturation", "gamma",
              "sharpness", "exposure"

        Returns
        -------
        int
            The current setting.

        """
        # Check if property exists
        prop = prop.casefold()
        if not prop in self.codes.keys():
            raise Exception("No such property!")
        
        if not self.camera.isOpened():
            raise Exception("Connection to camera was closed!")
        value = self.camera.get(self.codes[prop])
        
        # Properties can only have signed interger values
        return int(value)
         
    def setProperty(self, prop, value):
        """
        Sets one camera setting, as percent of the possible range.

        Parameters
        ----------
        prop : str
            Setting to change.
            Possible values are: 
              "brightness", "contrast", "hue", "saturation", "gamma",
              "sharpness", "exposure"
        value : float
            Should be in the range 0 to 1.

        Returns
        -------
        None.

        """
        # TODO: Check value here to give clearer error?
        prop = prop.casefold()
        value = self.ranges[prop][0] + (self.ranges[prop][1]-self.ranges[prop][0]) * value
        value = round(value)
        self.setPropertyNative(prop, value)
    
    def getProperty(self, prop):
        """
        Returns one camera setting, as percent of the possible range.

        Parameters
        ----------
        prop : str
            Setting to return.
            Possible values are: 
              "brightness", "contrast", "hue", "saturation", "gamma",
              "sharpness", "exposure"

        Returns
        -------
        value : float
            Current value in the range 0 to 1.

        """
        prop = prop.casefold()
        value = self.getPropertyNative(prop)
        value = (value - self.ranges[prop][0])/(self.ranges[prop][1]-self.ranges[prop][0])
        
        return value
    
    def getAllProperties(self):
        """
        Returns a dictionary with the values of all settings.

        Returns
        -------
        props : dict
            Current settings.

        """
        props = {}
        for p in self.codes.keys():
            props.update({p: self.getProperty(p)})
            
        return props
    
    def setAllProperties(self, props):
        """
        Sets all settings from a dictionary.

        Parameters
        ----------
        props : dict
            Should contain 'Setting':'Value' pairs. Value is between 0 and 1.
            Only tested when all settings are present.

        Returns
        -------
        None.

        """
        for p, v in props.items():
            self.setProperty(p, v)