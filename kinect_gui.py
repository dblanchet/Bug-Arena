import kinect
import numpy

import pygtk
pygtk.require('2.0')
import gtk

import gobject

import cairo

import time

GAMING_AREA = (-150.0, 100.0, 150.0, 300.0)  # Centimeters (x0, z0, x1, z1)
GAMING_DETECTION_ZONE = (37, 196, 566, 85)  # Pixels


class KinectDisplay(gtk.DrawingArea):

    def __init__(self):

        gtk.DrawingArea.__init__(self)
        self.set_size_request(1280, 480)

        self._found = False
        self._rgb_surface = None
        self._depth_surface = None

        self._observers = []

        self._x = -1
        self._y = -1
        self._obstacles = []
        self.refresh_data()

        self.add_events(gtk.gdk.MOTION_NOTIFY
                | gtk.gdk.BUTTON_PRESS
                | gtk.gdk.LEAVE_NOTIFY
                | gtk.gdk.LEAVE_NOTIFY_MASK)
        self.connect("motion_notify_event", self.motion_notify)
        self.connect("leave_notify_event", self.leave_notify)

        self.connect("expose_event", self.expose)

    def add_observer(self, observer):
        self._observers.append(observer)

    def _notify_observers(self):
        data = {}
        data['cursor'] = self._x, self._y, self._depth[self._y, self._x]
        data['obstacles'] = self._obstacles

        for observer in self._observers:
            observer.observable_changed(data)

    def leave_notify(self, widget, event):
        self._x, self._y = -1, -1
        self._notify_observers()
        self.queue_draw()

    def motion_notify(self, widget, event):
        x, y = event.x, event.y

        if x >= 640:
            x -= 640

        self._x, self._y = x, y
        self._notify_observers()
        self.queue_draw()

    def expose(self, widget, event):
        self.context = widget.window.cairo_create()
        self.draw(self.context)
        return False

    def refresh_data(self):
        # Get raw data.
        self._found_kinect, self._rgb, self._depth = kinect.get_buffers()

        # Perform basic data extraction.
        self._obstacles = kinect.extract_obstacles(
                self._depth,
                band=GAMING_DETECTION_ZONE,
                provide_raw=True)

        # Convert numpy arrays to cairo surfaces.
        alpha_channel = numpy.ones((480, 640, 1), dtype=numpy.uint8) * 255

        # 1. RGB bitmap.
        rgb32 = numpy.concatenate((alpha_channel, self._rgb), axis=2)
        self._rgb_surface = cairo.ImageSurface.create_for_data(
                rgb32[:, :, ::-1].astype(numpy.uint8),
                cairo.FORMAT_ARGB32, 640, 480)

        # 2. Depth map, take care of special NaN value.
        i = numpy.amin(self._depth)
        depth_clean = numpy.where(
                self._depth == kinect.UNDEF_DEPTH,
                0,
                self._depth)
        a = numpy.amax(depth_clean)
        depth = numpy.where(
                self._depth == kinect.UNDEF_DEPTH,
                0,
                255 - (self._depth - i) * 254.0 / (a - i))
        depth32 = numpy.dstack((
            alpha_channel, depth, numpy.where(depth == 0, 128, depth), depth))
        self._depth_surface = cairo.ImageSurface.create_for_data(
                depth32[:, :, ::-1].astype(numpy.uint8),
                cairo.FORMAT_ARGB32, 640, 480)

        self._notify_observers()

    def draw(self, ctx):

        # Draw surfaces.
        ctx.save()
        ctx.move_to(0, 0)
        ctx.set_source_surface(self._rgb_surface)
        ctx.paint()

        ctx.translate(640, 0)
        ctx.set_source_surface(self._depth_surface)
        ctx.paint()

        ctx.restore()

        # Dectection band.
        ctx.set_line_width(2)
        ctx.set_source_rgb(0.0, 0.0, 1.0)
        x, y, w, h = GAMING_DETECTION_ZONE
        x += 640
        ctx.rectangle(x, y, w, h)
        ctx.stroke()

        # Coordinate system.
        ctx.set_line_width(1)
        ctx.set_source_rgb(1.0, 1.0, 1.0)
        ctx.move_to(640 + 30, 470)
        ctx.line_to(640 + 10, 470)
        ctx.line_to(640 + 10, 450)
        ctx.stroke()

        ctx.select_font_face('Sans')
        ctx.set_font_size(12)
        ctx.move_to(640 + 3, 450)
        ctx.show_text('y')
        ctx.stroke()

        ctx.move_to(640 + 30, 477)
        ctx.show_text('x')
        ctx.stroke()

        # Trace lines.
        if self._x >= 0 and self._y >= 0:
            ctx.set_source_rgb(1.0, 0.0, 0.0)
            ctx.set_line_width(1)

            ctx.move_to(0, self._y)
            ctx.line_to(1280, self._y)
            ctx.stroke()

            ctx.move_to(self._x, 0)
            ctx.line_to(self._x, 480)
            ctx.stroke()

            ctx.move_to(self._x + 640, 0)
            ctx.line_to(self._x + 640, 480)
            ctx.stroke()

            # Tell about center_depth.
            depth = self._depth[self._y, self._x]
            distance = kinect.z_to_cm(depth)
            if distance != kinect.UNDEF_DISTANCE:
                text = "(%d, %d) - distance: %0.0f cm (depth = %d)" \
                        % (self._x, self._y, distance, depth)
            else:
                text = "(%d, %d)" % (self._x, self._y)

            ctx.set_font_size(16)
            ctx.move_to(950, 475)
            ctx.set_source_rgb(1, 1, 1)
            ctx.show_text(text)
            ctx.stroke()

        # Draw detected feet in detection zone.
        ctx.set_line_width(2)
        ctx.set_source_rgb(1, 0, 0)
        for obstacle in self._obstacles:
            raw_data = obstacle[-1]
            x, y, _ = raw_data[0]
            ctx.move_to(640 + x, y)
            for x, y, _ in raw_data[1:]:
                ctx.line_to(640 + x, y)
            ctx.stroke()

        # Tell if images are not from a present device.
        if not self._found_kinect:
            ctx.set_font_size(20)
            ctx.move_to(20, 20)
            ctx.set_source_rgb(0.0, 0.0, 1.0)
            ctx.show_text("No Kinect detected, using static picture from disk")
            ctx.stroke()


class GameSceneArea(gtk.DrawingArea):

    def __init__(self, size):
        gtk.DrawingArea.__init__(self)
        self.set_size_request(*size)
        self.connect("expose_event", self.expose)

        self._z = -1
        self._y = -1
        self._x = -1
        self._obstacles = []

        # Compute pixel per cm.
        l, h = size  # pixels
        x0, z0, x1, z1 = GAMING_AREA  # Centimeters
        x_ratio = l / (x1 - x0)
        y_ratio = h / max(z1, z0)
        self._px_per_cm = min(x_ratio, y_ratio)

    def expose(self, widget, event):
        self.context = widget.window.cairo_create()
        self.draw(self.context)
        return False

    def observable_changed(self, data):
        self._x, self._y, self._z = data['cursor']
        self._obstacles = data['obstacles']
        self.queue_draw()

    def draw(self, ctx):

        # Coordinate system.
        ctx.set_line_width(1)
        ctx.set_source_rgb(0.0, 0.0, 0.0)
        ctx.move_to(30, 470)
        ctx.line_to(10, 470)
        ctx.line_to(10, 450)
        ctx.stroke()

        ctx.select_font_face('Sans')
        ctx.set_font_size(12)
        ctx.move_to(3, 450)
        ctx.show_text('z')
        ctx.stroke()

        ctx.move_to(30, 477)
        ctx.show_text('x')
        ctx.stroke()

        # Kinect detection cone.
        ctx.set_line_width(.5)
        ctx.set_source_rgb(0.0, 0.0, 0.0)

        ctx.move_to(320, 479)
        ctx.line_to(0, 0)
        ctx.stroke()

        ctx.move_to(320, 479)
        ctx.line_to(640, 0)
        ctx.stroke()

        # Gaming zone.
        ctx.set_line_width(2)
        ctx.set_source_rgb(0.0, 0.0, 1.0)
        x0, z0, x1, z1 = GAMING_AREA  # Centimeters
        px = self.x_to_pixel(x0)
        py = self.z_to_pixel(z1)
        pw = self.x_to_pixel(x1) - px
        ph = self.z_to_pixel(z0) - py
        ctx.rectangle(px, py, pw, ph)
        ctx.stroke()

        # Current cursor depth.
        z = kinect.z_to_cm(self._z)
        if z0 < z < z1:

            # Draw line.
            ctx.set_line_width(1)
            ctx.set_source_rgb(1.0, 0.0, 0.0)
            py = self.z_to_pixel(z)
            ctx.move_to(0, py)
            ctx.line_to(640, py)
            ctx.stroke()

            x = - kinect.x_to_cm(self._x, z)
            px = self.x_to_pixel(x)
            ctx.move_to(px, py - 5)
            ctx.line_to(px, py - 10)
            ctx.stroke()
            ctx.move_to(px, py + 5)
            ctx.line_to(px, py + 10)
            ctx.stroke()

            # Add distance info.
            ctx.set_line_width(0.5)
            ctx.move_to(60, py)
            ctx.line_to(60, 480)
            ctx.stroke()

            ctx.set_font_size(16)
            ctx.move_to(50, 440)
            ctx.show_text('z')
            ctx.stroke()

            ctx.move_to(500, 440)
            ctx.show_text('x = %2.2f m' % (
                kinect.x_to_cm(self._x, z) / 100.0))
            ctx.stroke()

            ctx.move_to(500, 460)
            ctx.show_text('y = %2.2f m' % (
                kinect.y_to_cm(self._y, z) / 100.0))
            ctx.stroke()

            ctx.move_to(500, 480)
            ctx.show_text('z = %2.2f m' % (z / 100.0))
            ctx.stroke()

        # Detected feet.
        ctx.set_line_width(2)
        ctx.set_source_rgb(0.5, 0, 0)
        for obstacle in self._obstacles:
            x, y, w, h, z, raw_data = obstacle

            # Obstacle box.
            px = self.x_to_pixel(-x - w)
            py = self.z_to_pixel(y)
            pw = self.x_to_pixel(- x) - px
            ph = self.z_to_pixel(y + h) - py

            ctx.rectangle(px, py, pw, ph)
            ctx.set_line_width(2)
            ctx.set_source_rgb(0.0, 0.5, 0.0)
            ctx.stroke()

            # Raw data from Kinect.
            px, _, z = raw_data[0]
            x = self.x_to_pixel(-kinect.x_to_cm(px, z))
            y = self.z_to_pixel(z)
            ctx.move_to(x, y)
            for px, _, z in raw_data[1:]:
                x = self.x_to_pixel(-kinect.x_to_cm(px, z))
                y = self.z_to_pixel(z)
                ctx.line_to(x, y)
            ctx.set_line_width(0.5)
            ctx.set_source_rgb(0.7, 0.0, 0.0)
            ctx.stroke()

    def z_to_pixel(self, z):
        return 480 - int(z * self._px_per_cm)

    def x_to_pixel(self, x):
        return int(x * self._px_per_cm) + 320


class KinectTestWindow(gtk.Window):

    DATA_DIR = 'data/'
    REFRESH_DELAY = 500  # ms

    def __init__(self):
        self._paused = True

        gtk.Window.__init__(self)
        self.set_default_size(1280, 960)

        vbox = gtk.VBox()
        self.add(vbox)

        # Kinect info visualisation.
        self._display = KinectDisplay()
        vbox.pack_start(self._display, True, True, 0)

        hbox = gtk.HBox()
        vbox.pack_start(hbox)

        # Game scheme representation.
        game_scene = GameSceneArea((640, 480))
        self._display.add_observer(game_scene)
        hbox.pack_start(game_scene)

        button_vbox = gtk.VBox()
        hbox.pack_start(button_vbox)

        # Choose static data.
        self.choose = gtk.Button('Open', gtk.STOCK_OPEN)
        button_vbox.pack_start(self.choose)
        self.choose.connect("clicked", self._choose_cb)

        # Save button.
        self.save = gtk.Button('Save', gtk.STOCK_SAVE)
        self.save.set_sensitive(False)
        button_vbox.pack_start(self.save)
        self.save.connect("clicked", self._save_cb)

        # Pause/Autorefresh button.
        self.pause = gtk.Button('Pause', gtk.STOCK_MEDIA_PAUSE)
        button_vbox.pack_start(self.pause)
        self.pause.connect("clicked", self._pause_cb)

        self.connect("destroy", gtk.main_quit)
        self.show_all()

        # Auto-refresh at 10 frames per seconds.
        self.timer_id = gobject.timeout_add(self.REFRESH_DELAY,
                self._timedout)

    def _choose_cb(self, widget, data=None):
        # Create file chooser.
        dialog = gtk.FileChooserDialog("Open...",
                None,
                gtk.FILE_CHOOSER_ACTION_OPEN,
                (gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL,
                    gtk.STOCK_OPEN, gtk.RESPONSE_OK))
        dialog.set_default_response(gtk.RESPONSE_OK)
        dialog.set_current_folder(self.DATA_DIR)

        # Get only numpy arrays.
        filter = gtk.FileFilter()
        filter.set_name("Numpy arrays")
        filter.add_pattern("*_depth.npy")
        dialog.add_filter(filter)

        response = dialog.run()
        chosen = response == gtk.RESPONSE_OK
        if chosen:
            # Extract file basename.
            filename = dialog.get_filename()[:-10]
            kinect.set_default_data(filename)
            print filename, 'selected'

        dialog.destroy()

        # Refresh GUI if needed.
        if chosen:
            self._display.refresh_data()
            self.queue_draw()

    def _save_cb(self, widget, data=None):
        rgb = self._kinect.latest_rgb
        depth = self._kinect.latest_depth
        fname_base = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
        numpy.save(self.DATA_DIR + fname_base + '_rgb', rgb)
        numpy.save(self.DATA_DIR + fname_base + '_depth', depth)
        print 'Saved with "%s" base filename' % fname_base

    def _pause_cb(self, widget, data=None):
        self._paused = not self._paused
        self.save.set_sensitive(self._paused)
        self.choose.set_sensitive(self._paused)

        if not self._paused:
            self.pause.set_label(gtk.STOCK_MEDIA_PAUSE)
            # Try to prevent unwanted redraw.
            if not data:
                self._display.refresh_data()
                self.queue_draw()
            self.timer_id = gobject.timeout_add(self.REFRESH_DELAY,
                    self._timedout)
        else:
            self.pause.set_label(gtk.STOCK_REFRESH)

    def _timedout(self):
        # Stop auto refresh if no Kinect is detected.
        found_kinect, _, _ = kinect.get_buffers()
        if found_kinect:
            self._display.refresh_data()
            self.queue_draw()
        else:
            if not self._paused:
                print 'No Kinect found, stopping auto-refresh'
                self._pause_cb(None, True)

        # Timer is repeated until False is returned.
        return not self._paused

    def run(self):
        gtk.main()


def main():
    KinectTestWindow().run()

if __name__ == "__main__":
    main()
