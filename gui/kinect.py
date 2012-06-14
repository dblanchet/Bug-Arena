"""
Kinect.py

Interface to the kinect hardware.

"""

from collections import namedtuple
import numpy

try:
    import freenect
except ImportError:
    freenect = None
    print "Kinect module not found. Faking it"

__all__ = ['get_buffers', 'set_default_data', 'z_to_cm', 'x_to_cm', 'y_to_cm',
'extract_obstacles', 'get_obstacles',
'_UNDEF_DEPTH', '_UNDEF_DISTANCE', '_MIN_DISTANCE', '_MAX_DISTANCE',
]

_DEFAULT_ANALYSIS_BAND = (37, 196, 566, 85)
_DEFAULT_SURFACE = (-9999, -9999, 9999, 9999)


_UNDEF_DEPTH = 2047
_UNDEF_DISTANCE = 2000.0


# Look up table for depth calculations
# Formula from http://vvvv.org/forum/the-kinect-thread.
_dist_values = numpy.tan(numpy.arange(2048) / 1024.0 + 0.5) * 33.825 + 5.7

# XBox 360 Kinect is said to be OK with
# depth values between 80 cm and 4 meters.
# saturate all inputs/outputs to those values

_MIN_DISTANCE = 80.0  # cm
_MAX_DISTANCE = 400.0  # cm

_DIST_ARRAY = numpy.where(
        _MIN_DISTANCE < _dist_values,
        _dist_values,
        _UNDEF_DISTANCE)
_DIST_ARRAY = numpy.where(
        _DIST_ARRAY < _MAX_DISTANCE,
        _DIST_ARRAY,
        _UNDEF_DISTANCE)


# ----------------------------------------------
# Returned by get_buffers
KinectData = namedtuple('KinectData', 'real_kinect rgb depth')


def get_buffers():
    '''get_buffers(): returns a KinectData object
    KinectData members:
     - real_kinect (boolean) (true if data comes fro ma real kinect)
     - rgb array
     - depth array

     (buffers=numpy array)

     the input is taken from a file if the kinect is missing or the library not present.
     no memorization is done
     '''
    found_kinect = False

    if freenect:  # module has been imported
        try:
            # Try to obtain Kinect images.
            (depth, _), (rgb, _) = freenect.sync_get_depth(), freenect.sync_get_video()
            found_kinect = True
        except TypeError:
            pass

    if found_kinect:
        return KinectData(real_kinect=True, rgb=rgb, depth=depth)
    else:
        # Use local data files. not defined if not initialized
        return _DEFAULT_DATA


def set_default_data(filename):
    '''Sets default fake input file to use, without extension
     ex: 2012-03-02_14-36-48'''
    global _DEFAULT_FILE, _DEFAULT_DATA
    print "loaded fake data %s" % filename
    _DEFAULT_FILE = filename
    _DEFAULT_DATA = KinectData(
        real_kinect=False,
        rgb=numpy.load(filename + '_rgb.npy'),
        depth=numpy.load(filename + '_depth.npy')
        )


def z_to_cm(depth):
    "from a depth (or depth buffer), convert to depth in centimeters"
    return _DIST_ARRAY[depth]


def x_to_cm(x, z):
    "from a depth and x, converts to x in centimeters"
    coeff = 0.1734  # Measured constant.
    return (320.0 - x) * z * coeff


def y_to_cm(y, z):
    'from a depth and y, converts to y height in centimeters'
    coeff = 0.1734  # Measured constant.
    dev = 9 / coeff / 200  # Horizon is not at y = 0.
    h = 6.0  # Kinect captor is not at y = 0.
    return ((480.0 - y) - 240.0 - dev) * z * coeff + h




_Obstacle = namedtuple('Obstacle', 'x y width height z raw_data')
class Obstacle (_Obstacle):
    '''Returned by analyzer object.
    x, y : position related to front/center in cm Y is the depth
    w, h : size in cm
    height    Minimal y value detected in the obstacle. Int
    raw_data      Detected data. Numpy Array'''

    def __str__(self):
        return "Obstacle at (%.1f,%.1f) size : (%.1fx%.1f), height:%.1f %s" % (
            self.x, self.y,
            self.width, self.height,
            self.z,
             "(has raw data)" if self.raw_data else '(no raw data)'
             )


def extract_obstacles(depth, band=_DEFAULT_ANALYSIS_BAND, surface=_DEFAULT_SURFACE, provide_raw=False):
    '''Returns obstacles from pixel depth
    extract_obstacles(depth, band=..., surface=..., provide_raw=False):
        depth: depth array
        band: an optional analysis band in pixels (x, y, w, h) and
        surface: an optional analysis band in cm within the game area (x, z, w, p) - in top view, z is depth
        provide_raw : whether to provide raw data in the returned object or None

        returns a list of Obstacles objects

        Obstacle objects members:
             x:
             y:
             width:
             height:
                 coordinates of the bounding rectangle in top view *in centimeters*
                 (0,0) : center in front of kinect
             z: minimal height of the rectangle from the ground 0 => on the ground

             raw_data: the raw data for analysis (x,y in pixels, z in cm)
    '''
    MAX_DEPTH = 300.0  # 3 meters. FIXME Depends on Gaming Zone size.
    MAX_BORDER_HEIGHT = 10  # pixels. a foot can never be higher than this. Restrict accordingly
    MAX_Z_CHANGE = 10  # cm. consider discutinued foot if Z varies this much or more

    dist = z_to_cm(depth)

    # -- Extract borders (lower Y where Z is in range)
    bx, by, bw, bh = band

    borders = []     # list of (x, ymax, z@ymax) of non-empty columns. ymax : max Y where z is not null
    # x,y in pixels ; z in cm

    zone = dist[by:by + bh, bx:bx + bw]   # extract zone from which data is considered

    # ymax: for each x: maximum Y for the given X on the zone where Z is in range
    for x in xrange(zone.shape[1]):
        non_null_y = numpy.argwhere(zone[:, x] <= MAX_DEPTH)  # y in range ?
        if non_null_y.size:  # is there any z in the range ?
            ymax = numpy.max(non_null_y)
            # split to new if discontinuity (y ou z) ?
            borders.append((bx + x, by + ymax, zone[ymax, x]))
        # else: new foot

    # -- Analysis :

    # Analyze from the borders array the list of feets

    feet = []  # foot : (x,y,z) points in the foot, one per X
    x, _, z = borders[0]  # initialization
    prev_x = x
    prev_z = z
    foot = []  # current foot
    for x, y, z in borders:
        # Separate disconnected feet.
        # connected foot : contiguous X and not too abrupt z change
        if x - prev_x <= 1 and abs(prev_z - z) < MAX_Z_CHANGE:
            foot.append((x, y, z))
        else:
            feet.append(foot)
            foot = [(x, y, z)]
        prev_x = x
        prev_z = z

    if foot:
        feet.append(foot)

    # Limit zone height : distance between base and top must be restricted. shrink foot accordingly (...)
    # put back results to feet
    result = []
    for foot in feet:
        m = max(y_to_cm(y, z) for x, y, z in foot)  # top du pied actuel,  including corresponding z
        result.append([(x, y, z) for x, y, z in foot
            if m - y_to_cm(y, z) <= MAX_BORDER_HEIGHT])
    feet = result

    final = []
    for foot in feet:
        left = min(x_to_cm(x, z) for x, y, z in foot)
        right = max(x_to_cm(x, z) for x, y, z in foot)

        # z is already in cm
        close = min(z for x, y, z in foot)
        far = max(z for x, y, z in foot)

        bottom = min(y_to_cm(y, z) for x, y, z in foot)

        # swap coordinates Y and Z here (y was height, becomes depth ; invert for z)
        final.append(Obstacle(
            x=left,
            y=close,
            width=right - left,
            height=far - close,
            z=bottom,
            raw_data=foot if provide_raw else None
        ))
    return final


def get_obstacles(provide_raw=False):
    "get buffers from the Kinect and extract obstacles. See extract_obstacles and obstacle objects help for obstacle definition"
    k = get_buffers()
    if not k.real_kinect:
        print "Using Fake Data..."
    return extract_obstacles(k.depth, provide_raw=provide_raw)

set_default_data('2012-03-02_14-36-48')

if __name__ == '__main__':
    "test the library, don't execute if imported"
    for f in ('2012-03-02_14-36-48', '2012-03-23_12-55-38', '2012-03-23_13-21-48', '2012-03-30_14-14-43'):
        set_default_data(f)
        print 'Testing Kinect library ...'
        for foot in get_obstacles():
            print foot
