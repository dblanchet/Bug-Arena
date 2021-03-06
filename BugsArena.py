#
# cocos2d
# http://cocos2d.org
#
import random
import math

from cocos.director import director
from cocos.layer import Layer, ColorLayer
from cocos.scene import Scene
from cocos.scenes.transitions import RotoZoomTransition
from cocos.actions import RotateBy, Repeat, Reverse
from cocos.sprite import Sprite
from cocos import collision_model
from cocos import euclid

import pyglet
from pyglet.window import key

from cshape import OrientableRectShape


class HomeLayer(Layer):
    ''' Game menu. '''

    is_event_handler = True     # Enable pyglet's events

    def __init__(self):
        super(HomeLayer, self).__init__()
        screen_width, screen_height = director.get_window_size()
        self.text_title = pyglet.text.Label("Play",
            font_size=32,
            x = screen_width / 2,
            y = screen_height / 2,
            anchor_x='center',
            anchor_y='center')

    def draw(self):
        self.text_title.draw()

    def on_key_press(self, k, m):
        if k == key.ENTER:
            director.replace(RotoZoomTransition((gameScene), 1.25))
            return True
        else:
            return False


class BugLayer(Layer):
    ''' Layer on which the bugs walk. '''

    is_event_handler = True     # Enable pyglet's events.

    def __init__(self):
        super(BugLayer, self).__init__()
        self.schedule_interval(create_bug, 1)  # Creates a bug every 3 seconds

        cell_width = 100    # ~ bug image width * 1,25
        cell_height = 190   # ~bug image height * 1.25
        screen_width, screen_height = director.get_window_size()
        self.collision_manager = collision_model.CollisionManagerGrid(
                                                        0.0, screen_width,
                                                        0.0, screen_height,
                                                        cell_width,
                                                        cell_height)
        self.schedule(update, update)

    def on_mouse_press(self, x, y, buttons, modifiers):
        ''' invoked when the mouse button is pressed
            x, y the coordinates of the clicked point
        '''
        mouse_x, mouse_y = director.get_virtual_coordinates(x, y)

        for bug in self.collision_manager.objs_touching_point(mouse_x,
                                                              mouse_y):
            kill_bug(bug)


class Bug(Sprite):
    ''' Characters to be destroyed. '''

    def __init__(self):
        self.duration = random.randint(2, 8)
        if(self.duration < 5):
            image = 'bug1-small.png'
        else:
            image = 'bug2-small.png'

        bugSpriteSheet = pyglet.resource.image(image)
        bugGrid = pyglet.image.ImageGrid(bugSpriteSheet, 1, 6)
        animation_period = max(self.duration / 100, 0.05)  # seconds

        animation = bugGrid.get_animation(animation_period)
        super(Bug, self).__init__(animation)

        screen_height = director.get_window_size()[1]

        rect = self.get_rect()

        self.speed = (screen_height + rect.height) / self.duration
        self.cshape = OrientableRectShape(
            euclid.Vector2(rect.center[0], rect.center[1]),
                           rect.width / 2, rect.height / 2, 0)

    def start(self):
        ''' places the bug to its start position
            duration: the time in second for the
            bug to go to the bottom of the screen '''
        self.spawn()

        self.is_colliding = self.respawn_on_collision()
        collision_counter = 1 if self.is_colliding else 0
        while self.is_colliding:
            self.is_colliding = self.respawn_on_collision()
            if self.is_colliding:
                collision_counter += 1
            if collision_counter > 3:
                break

        self.rotation = -self.duration
        rotate = RotateBy(self.duration * 2, 1)
        self.do(Repeat(rotate + Reverse(rotate)))

        rect = self.get_rect()
        self.cshape.center = euclid.Vector2(rect.center[0], rect.center[1])

    def spawn(self):
        ''' pops at a random place on top of the screen '''
        rect = self.get_rect()
        half_width = rect.width / 2
        screen_width, screen_height = director.get_window_size()
        spawnX = random.randint(half_width, screen_width - half_width)
        self.position = (spawnX, screen_height + rect.height / 2)
        self.cshape.center.x, self.cshape.center.y = self.position
        self.cshape.update_position()
        self.cshape.rotate(self.rotation)

    def respawn_on_collision(self):
        ''' if the bug is colliding with another one,
            redraw it at another random place on top of screen
            returns True when it was colliding
        '''
        is_colliding = False
        for other in active_bug_list:
            if bugLayer.collision_manager.they_collide(self, other):
                self.spawn()
                is_colliding = True
                break
        return is_colliding

    def move_by(self, dx, dy):
        ''' moves the bug
            dx distance in pixels along horizontal axis
            dy distance in pixels along vertical axis '''
        self.position = (self.position[0] + dx, self.position[1] + dy)
        self.cshape.center.x, self.cshape.center.y = self.position
        self.cshape.update_position()
        self.cshape.rotate(self.rotation)


def update(dt, *args, **kwargs):
    ''' Updates the bugs position
        kills them when they are going out of the screen.
        invoked at each frame '''
    bugLayer.collision_manager.clear()
    for bug in active_bug_list:
        bugLayer.collision_manager.add(bug)

    screen_width = director.get_window_size()[0]

    for bug in active_bug_list:
        dy = bug.speed * dt / bug.duration
        if dy < 0.2:
            dy = 0.2
        can_move = True
        for other in bugLayer.collision_manager.iter_colliding(bug):
            if bug.get_rect().top >= other.get_rect().top:
                can_move = False    # blocked by a colliding bug

        if can_move:
            if bug.rotation > 180:
                rotation = bug.rotation - 360
            else:
                rotation = bug.rotation

            dx = - dy * math.sin(rotation)
            if bug.x - bug.width < 0 and dx < 0:
                dx = 0
            elif bug.x + bug.width > screen_width and dx > 0:
                dx = 0

            bug.move_by(dx, -dy)
            
            if bug.position[1] < 0:
                kill_bug(bug)


def create_bug(dt, *args, **kwargs):
    ''' Get a bug instance from the pool or
        creates one when the pool is empty. '''
    if len(bug_pool):
        bug = bug_pool.pop(random.randint(0, len(bug_pool) - 1))
    else:
        bug = Bug()
    bugLayer.add(bug)
    bugLayer.collision_manager.add(bug)
    bug.start()
    active_bug_list.append(bug)
    if bug.is_colliding:
        kill_bug(bug) # the bug did not find any free place to spawn

def kill_bug(bug):
    active_bug_list.remove(bug)
    bug.stop()
    try:
        bugLayer.collision_manager.remove_tricky(bug)
    except:
        pass
    bugLayer.remove(bug)
    bug_pool.append(bug)

if __name__ == "__main__":

    pyglet.resource.path = ['images', 'sounds', 'fonts']
    pyglet.resource.reindex()

    director.init(resizable=True)
    director.window.set_fullscreen(False)

    active_bug_list = []
    bug_pool = []
    for i in range(50):
        bug_pool.append(Bug())

    homeLayer = HomeLayer()
    colorLayer = ColorLayer(128, 16, 16, 255)
    homeScene = Scene(colorLayer, homeLayer)

    bugLayer = BugLayer()
    gameScene = Scene(bugLayer)

    director.run(homeScene)
