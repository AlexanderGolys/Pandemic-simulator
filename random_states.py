import random
from enum import Enum, auto
import time
import copy

import numpy as np
import turtle as tt

from main import Data


def positive_int_check(constant, name):
    if not isinstance(constant, int) or constant < 1:
        raise TypeError(f"{name} must be a positive number instead of {constant}.")


class GlobalSetup:
    """
    About:
        Application works with practically any population size (with large samples being computed not in real time
        to make animation more fluent. With really big samples, like 500+, the drawing itself can be laggy.

        Firstly the ball's position is initialized, and the process can take a while when the screen resolution is big
        as well as sample size. The progress is printed to the console every 10% of the way to success.

        Then, if population size is bigger than 70, the simulation is computed. Mostly this process is much faster than
        balls initialization, so there is no progress printed. Also it's hard to evaluate it during simulation.

        Thirdly, the animation is run (if the population size is less or equal 70, simulation and animation are made
        in the same time). You can choose preferred FPS, but it will not be real frames per second - the drawing
        process is quite slow and it's duration is hard to evaluate, thus also hard is fitting correct sleep time to
        maintain correct real FPS value.

        There is the option to change constant recovery time, noted as RECOVERY_TIME or normally distributed recovery
        time, where RECOVERY_TIME is expected value and RECOVERY_STANDARD_DEVIATION is sigma. If recovery time is drawn
        to be negative, the particle behaves as it would be 0.

        The last stage of simulation is printing the plot of healthy, infected and recovered particles over time
        and analysing gathered data using differential equations from SIR model, next printing them in stats in
        section below the plot.

        More on https://github.com/AlexanderGolys/pandemic-simulator/.

    Constants:
        BALL_RADIUS: Radius of every particle in pixels.
        BALL_SPEED: Norm of initial velocity vector.
        FPS: Frames per second (not realistic).
        NO_BALLS: Number of particles in simulation.
        SOCIAL_DISTANCING: Fraction of population that is not moving.
        RECOVERY_TIME: Expected value of recovery time.
        WINDOW_RESOLUTION: Window resolution.
        ITERATION: Maximal number of iteration. Program will be executed earlier, if there were no infected particles.
        DIVISION_THRESHOLD: After this number of iterations jammed particles will try to separate.
        COLLISION_THRESHOLD: Particles will bounce when distance between them is smaller than this value.
        RANDOM_RECOVERY_TIME: If True, each particle's recovery time is distributed from
                                N(RECOVERY_TIME, RECOVERY_STANDARD_DEVIATION**2)
                              If False, each particle's recovery time is constant and equal RECOVERY_TIME.
        RECOVERY_STANDARD_DEVIATION: Parameter of recovery time distribution in case of RANDOM_RECOVERY_TIME being True.
    """

    BALL_RADIUS = 5
    positive_int_check(BALL_RADIUS, "Radius")

    FPS = 30
    positive_int_check(FPS, "FPS")

    NO_BALLS = 500
    positive_int_check(NO_BALLS, "Number of particles")

    RECOVERY_PROB = .1

    WINDOW_RESOLUTION = (1200, 800)
    positive_int_check(WINDOW_RESOLUTION[0], "Window width")
    positive_int_check(WINDOW_RESOLUTION[1], "Window height")

    ITERATIONS = 10000
    positive_int_check(ITERATIONS, "Maximal number of iterations")


class Colors:
    """
    Colors used in this project.
    get_color is a dictionary matching color to corresponding particle state.
    """
    GREEN = (0x22, 0xEE, 0x22)
    RED = (0xEE, 0x22, 0x22)
    # YELLOW = (0xFF, 0x7F, 0)
    GRAY = (0xAA, 0xAA, 0xAA)

    BG_COLOR = (0xEE, 0xEE, 0xEE)
    WHITE = (0xFF, 0xFF, 0xFF)
    BLACK = (0, 0, 0)

    get_color = {
        States.HEALTHY: GREEN,
        States.INFECTED: RED,
        States.RECOVERED: GRAY
    }


class Ball:
    """
    Class representing particle.
    Attributes:
        center (tuple[int]): Coordinates of the center of the particle.
        x (int): First coordinate.
        y (int): Second coordinate.
        radius (int): Radius of the particle, by default value of GlobalSetup.BALL_RADIUS.
        state (key of States): Particle state (by default healthy).
        color (tuple[int]): Particle color, state dependent.


    Properties:
        center.getter: Returning center.
        center.setter: Setting center and x, y accordingly.
        state.getter: Returning state.
        state.setter: Setting state and changing color.

    Methods:
        __sub__(self, other): Calculating the distance between particles.
        draw(self): Drawing particle in real-time simulation (using its attributes)
        draw_offline(center, color): Draw particle with simulation already performed, based only on information
            about its color and position (static).


    """
    def __init__(self, corners_x, corners_y):
        self.corners_x = corners_x
        self.corners_y = corners_y
        self.center = (random.randint(*corners_x), random.randint(*corners_y))
        self.radius = GlobalSetup.BALL_RADIUS
        self.state = States.HEALTHY
        self.color = Colors.get_color[self.state]

    def random_pos(self):
        self.center = (random.randint(*self.corners_x), random.randint(*self.corners_y))

    @property
    def center(self):
        return self.__center

    @center.setter
    def center(self, new):
        self.__center = new
        self.x = new[0]
        self.y = new[1]

    @property
    def x(self):
        return self.center[0]

    @x.setter
    def x(self, value):
        self.__center = (value, self.center[1])

    @property
    def y(self):
        return self.center[1]

    @y.setter
    def y(self, value):
        self.__center = (self.center[0], value)

    @property
    def state(self):
        return self.__state

    @state.setter
    def state(self, state):
        self.__state = state
        self.color = Colors.get_color[self.__state]

    def __sub__(self, other):
        x1, y1 = self.center
        x2, y2 = other.center
        return ((x1 - x2)**2 + (y1 - y2)**2)**(1/2) - self.radius - other.radius

    def draw(self):
        tt.up()
        tt.setpos(Graphics.shifted(self.center))
        tt.dot(2*self.radius, self.color)

    @staticmethod
    def draw_offline(center, color):
        tt.up()
        tt.setpos(Graphics.shifted(center))
        tt.dot(2 * GlobalSetup.BALL_RADIUS, color)


class PopulationMethods:
    """
    Set of static methods operating on the population list.
    Methods:
        constructor(): Creating new population
        set_moving(population): Setting GlobalSetup.SOCIAL_DISTANCING% of population not to move.
        balls_collision(ball1, ball2): Simulate non-central elastic collision of 2 balls.
        update_population(population): changing each particle state in new iteration.
    """

    @staticmethod
    def constructor():
        population = []
        for i in range(GlobalSetup.NO_BALLS):
            ball = Ball((5*GlobalSetup.BALL_RADIUS, GlobalSetup.WINDOW_RESOLUTION[0]-5*GlobalSetup.BALL_RADIUS),
                        (5*GlobalSetup.BALL_RADIUS, GlobalSetup.WINDOW_RESOLUTION[1]-5*GlobalSetup.BALL_RADIUS))
            population.append(ball)

        population[-1].state = States.INFECTED
        print(f"Initialised {len(population)} samples.")
        return population

    @staticmethod
    def balls_collision(ball1, ball2):
        if ball1.state == States.INFECTED and ball2.state == States.HEALTHY:
            ball2.state = States.INFECTED
            Data.data[-1].infect()
        if ball1.state == States.HEALTHY and ball2.state == States.INFECTED:
            ball1.state = States.INFECTED
            Data.data[-1].infect()

    @staticmethod
    def update_population(population):
        ball_set = set(population)
        second_ball_set = set(population)
        for ball1 in ball_set:
            second_ball_set.difference_update({ball1})
            for ball2 in second_ball_set:
                if ball1 - ball2 <= 0:
                    PopulationMethods.balls_collision(ball1, ball2)

        for ball in population:
            ball.random_pos()
            if ball.state == States.INFECTED:
                if random.random() < GlobalSetup.RECOVERY_PROB:
                    ball.state = States.RECOVERED
                    Data.data[-1].recover()


class Graphics:
    """
    Set of static methods responsible for animation.

    Methods:
        init(): Set basic turtle parameters.
        write(string, center): Write given string on the screen.
        draw(population): Drawing the frame while performing simulation in real time.
        draw_offline(states): Drawing the frame while performing simulation before animation.
        shifted(coords): Returning shifted coords, such as (0, 0) is in the left-bottom corner of the screen.
        draw_rect(coords_a, coords_b, fill_color): Drawing rectangle with given corners and color.
        draw_line(a, b, color, pensize): Drawing line with given begin and end point.
        plot_line(jump, value_jump, w, h, what_to_plot): Method used for line plots, then replaced by area plots.
        plot_area(jump, y_axis, w, h): Drawing ending plot at stats screen.
        draw_stats(): Drawing stats screen.


    """
    @staticmethod
    def init():
        tt.setup(*GlobalSetup.WINDOW_RESOLUTION)
        tt.mode("logo")
        tt.colormode(255)
        tt.title("Pandemic Simulation")
        tt.tracer(0, 0)
        tt.hideturtle()

    @staticmethod
    def write(string, center):
        tt.up()
        tt.setpos(center)
        tt.write(string, align="left", font=("Cantarell", 14, "normal"))

    @staticmethod
    def draw(population):
        tt.reset()
        tt.hideturtle()
        for ball in population:
            ball.draw()
        tt.update()
        time.sleep(1/GlobalSetup.FPS)

    @staticmethod
    def offline_draw(states):
        tt.reset()
        tt.hideturtle()
        for state in states:
            Ball.draw_offline(*state)
        tt.update()
        time.sleep(1/(2*GlobalSetup.FPS))

    @staticmethod
    def shifted(coords):
        x, y = coords
        return x - GlobalSetup.WINDOW_RESOLUTION[0] // 2, y - GlobalSetup.WINDOW_RESOLUTION[1] // 2

    @staticmethod
    def draw_rect(coords_a, coords_b,  fill_color):
        a, b = coords_a
        c, d = coords_b
        tt.up()
        tt.setpos(coords_a)
        tt.down()
        tt.fillcolor(fill_color)
        tt.pencolor(fill_color)
        tt.begin_fill()
        tt.setpos(a, d)
        tt.setpos(coords_b)
        tt.setpos(c, b)
        tt.setpos(coords_a)
        tt.end_fill()
        tt.update()

    @staticmethod
    def draw_line(a, b, color=Colors.BLACK, pensize=3):
        tt.pensize(pensize)
        tt.pencolor(color)
        tt.up()
        tt.setpos(a)
        tt.down()
        tt.setpos(b)

    @staticmethod
    def plot_line(jump, value_jump, w, h, what_to_plot):
        current_x_index = 0
        prev_y = Data.data[current_x_index][what_to_plot] * value_jump + 11 * h // 20
        for current_x_pixel in range(w // 3 + h // 20 + 2, w * 2 // 3 - h // 20):
            current_x_index += jump
            current_y = Data.data[int(current_x_index)][what_to_plot] * value_jump + 11 * h // 20
            Graphics.draw_line(Graphics.shifted((current_x_pixel - 1, prev_y)),
                               Graphics.shifted((current_x_pixel, current_y)), Colors.RED, 2)
            prev_y = current_y

    @staticmethod
    def plot_area(jump, y_axis, w, h):
        current_x_index = 0
        for current_x_pixel in range(w // 3 + h // 20 + 2, w * 2 // 3 - h // 20):
            len1 = y_axis * Data.data[int(current_x_index)][States.HEALTHY] / GlobalSetup.NO_BALLS
            len2 = y_axis * Data.data[int(current_x_index)][States.INFECTED] / GlobalSetup.NO_BALLS
            len3 = y_axis * Data.data[int(current_x_index)][States.RECOVERED] / GlobalSetup.NO_BALLS
            Graphics.draw_line(Graphics.shifted((current_x_pixel, 11*h//20)),
                               Graphics.shifted((current_x_pixel, 11*h//20+len3)), Colors.GRAY, 1)
            Graphics.draw_line(Graphics.shifted((current_x_pixel, 11*h//20+len3)),
                               Graphics.shifted((current_x_pixel, 11*h//20+len3+len2)), Colors.RED, 1)
            Graphics.draw_line(Graphics.shifted((current_x_pixel, 11*h//20+len3+len2)),
                               Graphics.shifted((current_x_pixel, 11*h//20+len2+len3+len1)), Colors.GREEN, 1)
            current_x_index += jump

    @staticmethod
    def draw_stats():
        w, h = GlobalSetup.WINDOW_RESOLUTION

        # draw background
        Graphics.draw_rect(Graphics.shifted((0, 0)), Graphics.shifted((w, h)), Colors.BG_COLOR)
        Graphics.draw_rect(Graphics.shifted((w//3, h//2)), Graphics.shifted((w*2//3, h*9//10)), Colors.WHITE)
        Graphics.draw_rect(Graphics.shifted((w//3, h*2//5)), Graphics.shifted((w*2//3, 0)), Colors.WHITE)

        x_axis_len = w//3 - h//10 - 1
        y_axis_len = h*3//10
        jump = len(Data.data) / x_axis_len
        # value_jump = y_axis_len / GlobalSetup.no_balls
        Graphics.plot_area(jump, y_axis_len, w, h)

        # draw axes
        Graphics.draw_line(Graphics.shifted((w//3 + h//20, h*11//20)), Graphics.shifted((w//3 + h//20, h*17//20)))
        Graphics.draw_line(Graphics.shifted((w//3 + h//20, h*11//20)), Graphics.shifted((w*2//3 - h//20, h*11//20)))

        # draw upper arrow
        Graphics.draw_line(Graphics.shifted((w//3 + h//20, h*17//20)),
                           Graphics.shifted((w//3 + h//20 - h//200, h*17//20 - h//200)))
        Graphics.draw_line(Graphics.shifted((w//3 + h//20, h*17//20)),
                           Graphics.shifted((w//3 + h//20 + h//200, h*17//20 - h//200)))

        # draw right arrow
        Graphics.draw_line(Graphics.shifted((w*2//3 - h//20, h*11//20)),
                           Graphics.shifted((w*2//3 - h//20 - h//200, h*11//20 + h//200)))
        Graphics.draw_line(Graphics.shifted((w*2//3 - h//20, h*11//20)),
                           Graphics.shifted((w*2//3 - h//20 - h//200, h*11//20 - h//200)))

        shift = 25
        # Labels
        Graphics.write("Population size:", Graphics.shifted((w//3 + w//30, h*2//5 - 2*shift)))
        Graphics.write("Speed:", Graphics.shifted((w//3 + w//30, h*2//5 - 3*shift)))
        Graphics.write("Recovery:", Graphics.shifted((w//3 + w//30, h*2//5 - 4*shift)))
        Graphics.write("Social distancing:", Graphics.shifted((w//3 + w//30, h*2//5 - 5*shift)))
        Graphics.write("Ball radius:", Graphics.shifted((w//3 + w//30, h*2//5 - 6*shift)))

        Graphics.write("Infected:", Graphics.shifted((w//3 + w//30, h*2//5 - 8*shift)))
        Graphics.write("Duration:", Graphics.shifted((w//3 + w//30, h*2//5 - 9*shift)))
        Graphics.write("\u03B2 (SIR model):", Graphics.shifted((w//3 + w//30, h*2//5 - 10*shift)))
        Graphics.write("\u03B3 (SIR model):", Graphics.shifted((w//3 + w//30, h*2//5 - 11*shift)))
        Graphics.write("R0:", Graphics.shifted((w//3 + w//30, h*2//5 - 12*shift)))

        # Values
        Graphics.write(f"{GlobalSetup.NO_BALLS}", Graphics.shifted((w // 2, h * 2 // 5 - 2 * shift)))
        Graphics.write(f"No speed", Graphics.shifted((w // 2, h * 2 // 5 - 3 * shift)))
        Graphics.write(f"{GlobalSetup.RECOVERY_PROB} chance/iteration", Graphics.shifted((w//2, h*2//5 - 4*shift)))
        Graphics.write(f"No social distancing", Graphics.shifted((w // 2, h * 2 // 5 - 5 * shift)))
        Graphics.write(f"{GlobalSetup.BALL_RADIUS}", Graphics.shifted((w // 2, h * 2 // 5 - 6 * shift)))

        Graphics.write(f"{Data.data[-1][States.RECOVERED] / GlobalSetup.NO_BALLS * 100: .2f}% ({Data.data[-1][States.RECOVERED]})",
                       Graphics.shifted((w // 2, h * 2 // 5 - 8 * shift)))
        Graphics.write(f"{len(Data.data)} iterations", Graphics.shifted((w//2, h*2//5 - 9*shift)))
        beta, gamma, r0 = Data.SIR_analyse()
        beta2, gamma2, r02 = Data.SIR_analyse_improved()
        Graphics.write(f"{beta: .2f} ({beta2: .2f})", Graphics.shifted((w//2, h*2//5 - 10*shift)))
        Graphics.write(f"{gamma: .2f} ({gamma2: .2f})", Graphics.shifted((w//2, h*2//5 - 11*shift)))
        Graphics.write(f"{r0: .2f} ({r02: .2f})", Graphics.shifted((w//2, h*2//5 - 12*shift)))

        tt.update()
        tt.exitonclick()


def main():
    """
    Main loop.
    """
    population = PopulationMethods.constructor()
    Graphics.init()
    for i in range(GlobalSetup.ITERATIONS):
        Data.data.append(copy.deepcopy(Data.data[-1]))
        PopulationMethods.update_population(population)
        Graphics.draw(population)
        if Data.data[-1][States.INFECTED] == 0:
            break

    Graphics.draw_stats()


if __name__ == "__main__":
    main()
