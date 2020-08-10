import random
from enum import Enum, auto
import time
import warnings
import copy

import numpy as np
import turtle as tt


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

    BALL_RADIUS = 10
    positive_int_check(BALL_RADIUS, "Radius")

    BALL_SPEED = 300  # Speed, as an initial norm of velocity vector, doesn't have to be int, but has to be positive
    positive_int_check(int(BALL_SPEED), "Speed")

    FPS = 60
    positive_int_check(FPS, "FPS")

    NO_BALLS = 100
    positive_int_check(NO_BALLS, "Number of particles")

    SOCIAL_DISTANCING = .2
    if not 0 <= SOCIAL_DISTANCING <= 1:
        raise TypeError(f"Social distancing parameter must belong to <0, 1> instead of {SOCIAL_DISTANCING}.")

    RECOVERY_TIME = 300
    positive_int_check(RECOVERY_TIME, "Expected recovery time")

    WINDOW_RESOLUTION = (1600, 900)
    positive_int_check(WINDOW_RESOLUTION[0], "Window width")
    positive_int_check(WINDOW_RESOLUTION[1], "Window height")

    ITERATIONS = 10000
    positive_int_check(ITERATIONS, "Maximal number of iterations")

    DIVISION_THRESHOLD = 5
    positive_int_check(DIVISION_THRESHOLD, "Division threshold")

    COLLISION_THRESHOLD = 1
    if not isinstance(COLLISION_THRESHOLD, int) or COLLISION_THRESHOLD < 0:
        raise TypeError(f"Collision threshold must be a non-negative integer instead of {COLLISION_THRESHOLD}.")

    RANDOM_RECOVERY_TIME = True
    if not isinstance(RANDOM_RECOVERY_TIME, bool):
        raise TypeError(f"RANDOM_RECOVERY_TIME must be a bool instead of {RANDOM_RECOVERY_TIME}.")

    RECOVERY_STANDARD_DEVIATION = 30
    if RECOVERY_STANDARD_DEVIATION < 0:
        raise TypeError(f"Standard deviation must be non-negative instead of {RECOVERY_STANDARD_DEVIATION}.")


class Vector:
    """
    Class representing simple 2D vector.

    Methods:
        __add__(self, other): Standard vector addition.
        __mul__(self, other): Dot product.
        __sub__(self, other): Standard vector subtraction.
        len(self): Euclidean norm.
        neg(self): Multiplication by -1.
        up(self): changing y coordinate to be positive (bouncing off bottom wall)
        down(self): changing y coordinate to be negative (bouncing off top wall)
        left(self): changing x coordinate to be negative (bouncing off left wall)
        right(self): changing x coordinate to be positive (bouncing off right wall)
        normalise(self): Normalise itself.
        denormalise_to_speed(self): Increasing it's norm to the value of GlobalSetup.BALL_SPEED.
        shift(self, coords): Shifting coords with itself.
        constant_mul(self, c): Multiplication by number.

    """
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)

    def __mul__(self, other):
        return self.x * other.x + self.y * other.y

    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y)

    def len(self):
        return (self.x ** 2 + self.y ** 2)**(1/2)

    def neg(self):
        self.x *= -1
        self.y *= -1

    def up(self):
        self.y = abs(self.y)

    def down(self):
        self.y = -abs(self.y)

    def left(self):
        self.x = -abs(self.x)

    def right(self):
        self.x = abs(self.x)

    def normalise(self):
        self.x /= self.len()
        self.y /= self.len()

    def denormalise_to_speed(self):
        self.x *= (GlobalSetup.BALL_SPEED / GlobalSetup.FPS) ** (1 / 2)
        self.y *= (GlobalSetup.BALL_SPEED / GlobalSetup.FPS) ** (1 / 2)

    def shift(self, coords):
        return coords[0] + self.x, coords[1] + self.y

    def constant_mul(self, c):
        return Vector(self.x*c, self.y*c)


class States(Enum):
    """
    Enum class gathering possible particle states.
    """
    HEALTHY = auto()
    INFECTED = auto()
    RECOVERED = auto()


class Colors:
    """
    Colors used in this project.
    get_color is a dictionary matching color to corresponding particle state.
    """
    GREEN = (0x22, 0xEE, 0x22)
    RED = (0xEE, 0x22, 0x22)
    # YELLOW = (0xFF, 0x7F, 0)
    YELLOW = (0xAA, 0xAA, 0xAA)

    BG_COLOR = (0xEE, 0xEE, 0xEE)
    WHITE = (0xFF, 0xFF, 0xFF)
    BLACK = (0, 0, 0)

    get_color = {
        States.HEALTHY: GREEN,
        States.INFECTED: RED,
        States.RECOVERED: YELLOW
    }


class Ball:
    """
    Class representing particle.
    Attributes:
        center (tuple[int]): Coordinates of the center of the particle.
        x (int): First coordinate.
        y (int): Second coordinate.
        radius (int): Radius of the particle, by default value of GlobalSetup.BALL_RADIUS.
        moving (bool): Information if particle is moving.
        velocity (Vector): Velocity vector.
        state (key of States): Particle state (by default healthy).
        color (tuple[int]): Particle color, state dependent.
        infection_time: Number of iterations that particle is infected (if not infected: 0)
        collision_time: Number of iteration that particle is jammed.
        recovery_time (int): Recovery time.

    Properties:
        center.getter: Returning center.
        center.setter: Setting center and x, y accordingly.
        state.getter: Returning state.
        state.setter: Setting state and changing color.
        moving.getter: Returning moving.
        moving.setter: If moving is False, set velocity to the zero vector.

    Methods:
        __sub__(self, other): Calculating the distance between particles.
        draw(self): Drawing particle in real-time simulation (using its attributes)
        draw_offline(center, color): Draw particle with simulation already performed, based only on information
            about its color and position (static).


    """
    def __init__(self, av):
        self.center = list(random.choice(tuple(av)))
        self.x = self.center[0]
        self.y = self.center[1]
        self.radius = GlobalSetup.BALL_RADIUS
        self.moving = None
        self.velocity = Vector(random.uniform(-1, 1), random.uniform(-1, 1))
        self.velocity.normalise()
        self.velocity.denormalise_to_speed()
        self.state = States.HEALTHY
        self.color = Colors.get_color[self.state]
        self.infection_time = 0
        self.collision_time = 0
        self.collision = False
        self.recovery_time = GlobalSetup.RECOVERY_TIME
        if GlobalSetup.RANDOM_RECOVERY_TIME:
            self.recovery_time = random.gauss(GlobalSetup.RECOVERY_TIME, GlobalSetup.RECOVERY_STANDARD_DEVIATION)

    @property
    def center(self):
        return self.__center

    @center.setter
    def center(self, new):
        self.__center = new
        self.x = new[0]
        self.y = new[1]

    @property
    def state(self):
        return self.__state

    @state.setter
    def state(self, state):
        self.__state = state
        self.color = Colors.get_color[self.__state]

    @property
    def moving(self):
        return self.__moving

    @moving.setter
    def moving(self, other):
        self.__moving = other
        if not other:
            self.velocity = Vector(0, 0)

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
        if GlobalSetup.NO_BALLS > 70:
            warnings.warn("Number of balls bigger than 70: real time simulation is too computationally heavy, "
                          "so simulation and displaying will be performed separately.")
            time.sleep(1)
            Data.buffering_data = []
        print("Initialising balls...", flush=True)
        population = []
        available = {(i, j) for i in range(GlobalSetup.BALL_RADIUS,
                                           GlobalSetup.WINDOW_RESOLUTION[0] - GlobalSetup.BALL_RADIUS)
                     for j in range(GlobalSetup.BALL_RADIUS,
                                    GlobalSetup.WINDOW_RESOLUTION[1] - GlobalSetup.BALL_RADIUS)}
        for i in range(GlobalSetup.NO_BALLS):
            if i % (GlobalSetup.NO_BALLS // 10) == 0:
                print(f"{100*i//GlobalSetup.NO_BALLS}% ({i}/{GlobalSetup.NO_BALLS})")
            ball = Ball(available)
            population.append(ball)
            for c1 in range(ball.x - 2*ball.radius, ball.x + 2*ball.radius + 1):
                for c2 in range(ball.y - 2*ball.radius, ball.y + 2*ball.radius + 1):
                    available.difference_update({(c1, c2)})

        population[-1].state = States.INFECTED
        print(f"Initialised {len(population)} samples.")

        return population

    @staticmethod
    def set_moving(population):
        for i, ball in enumerate(population):
            ball.moving = i > GlobalSetup.SOCIAL_DISTANCING

    @staticmethod
    def balls_collision(ball1, ball2):
        v1 = ball1.velocity
        v2 = ball2.velocity
        if ball1.moving and ball2.moving:
            x_diff = Vector(ball1.x - ball2.x, ball1.y - ball2.y)
            ball1.velocity -= x_diff.constant_mul((v1 - v2) * x_diff / x_diff.len() ** 2)
            x_diff.neg()
            ball2.velocity -= x_diff.constant_mul((v2 - v1) * x_diff / x_diff.len() ** 2)
        elif ball1 - ball2 <= 2 and ball1.moving and not ball2.moving:
            x_diff = Vector(ball1.x - ball2.x, ball1.y - ball2.y)
            ball1.velocity -= x_diff.constant_mul(2 * ((v1 - v2) * x_diff / x_diff.len() ** 2))
        else:
            x_diff = Vector(ball2.x - ball1.x, ball2.y - ball1.y)
            ball2.velocity -= x_diff.constant_mul(2 * ((v2 - v1) * x_diff / x_diff.len() ** 2))
        if ball1.state == States.INFECTED and ball2.state == States.HEALTHY:
            ball2.state = States.INFECTED
            Data.data[-1].infect()
        if ball1.state == States.HEALTHY and ball2.state == States.INFECTED:
            ball1.state = States.INFECTED
            Data.data[-1].infect()

    @staticmethod
    def update_population(population):
        for ball in population:
            if ball.moving:
                ball.center = ball.velocity.shift(ball.center)
            if ball.state == States.INFECTED:
                ball.infection_time += 1
                if ball.infection_time > ball.recovery_time:
                    ball.state = States.RECOVERED
                    Data.data[-1].recover()
            if not ball.collision:
                ball.collision_time = 0
            ball.collision = False

        ball_set = set(population)
        second_ball_set = set(population)
        for ball1 in ball_set:
            second_ball_set.difference_update({ball1})
            for ball2 in second_ball_set:
                if ball1 - ball2 <= GlobalSetup.COLLISION_THRESHOLD:
                    PopulationMethods.balls_collision(ball1, ball2)
                    ball1.collision = True
                    ball2.collision = True
                    ball1.collision_time += 1
                    ball2.collision_time += 1
                    if ball1.collision_time > GlobalSetup.DIVISION_THRESHOLD \
                            and ball2.collision_time > GlobalSetup.DIVISION_THRESHOLD:
                        if ball1.velocity.len() > ball2.velocity.len():
                            shift = copy.deepcopy(ball1.velocity)
                            shift.normalise()
                            shift = shift.constant_mul(2*ball1.radius)
                            ball1.center = shift.shift(ball1.center)
                        else:
                            shift = copy.deepcopy(ball2.velocity)
                            shift.normalise()
                            shift = shift.constant_mul(2*ball2.radius)
                            ball2.center = shift.shift(ball2.center)
                        ball1.collision_time = 0
                        ball2.collision_time = 0

            if ball1.x - ball1.radius <= GlobalSetup.COLLISION_THRESHOLD:
                ball1.velocity.right()
            if ball1.x + ball1.radius >= GlobalSetup.WINDOW_RESOLUTION[0]-2:
                ball1.velocity.left()
            if ball1.y - ball1.radius <= GlobalSetup.COLLISION_THRESHOLD:
                ball1.velocity.up()
            if ball1.y + ball1.radius >= GlobalSetup.WINDOW_RESOLUTION[1]-2:
                ball1.velocity.down()


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
                               Graphics.shifted((current_x_pixel, 11*h//20+len3)), Colors.YELLOW, 1)
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
        Graphics.write(f"{GlobalSetup.BALL_SPEED}", Graphics.shifted((w // 2, h * 2 // 5 - 3 * shift)))
        if GlobalSetup.RANDOM_RECOVERY_TIME:
            Graphics.write(f"N({GlobalSetup.RECOVERY_TIME}, {GlobalSetup.RECOVERY_STANDARD_DEVIATION})",
                           Graphics.shifted((w // 2, h * 2 // 5 - 4 * shift)))
        else:
            Graphics.write(f"{GlobalSetup.RECOVERY_TIME}", Graphics.shifted((w//2, h*2//5 - 4*shift)))
        Graphics.write(f"{GlobalSetup.SOCIAL_DISTANCING}", Graphics.shifted((w // 2, h * 2 // 5 - 5 * shift)))
        Graphics.write(f"{GlobalSetup.BALL_RADIUS}", Graphics.shifted((w // 2, h * 2 // 5 - 6 * shift)))

        Graphics.write(f"{Data.data[-1][States.RECOVERED] / GlobalSetup.NO_BALLS * 100: .2f}%",
                       Graphics.shifted((w // 2, h * 2 // 5 - 8 * shift)))
        Graphics.write(f"{len(Data.data)} iterations", Graphics.shifted((w//2, h*2//5 - 9*shift)))
        beta, gamma, r0 = Data.SIR_analyse()
        Graphics.write(f"{beta: .4f}", Graphics.shifted((w//2, h*2//5 - 10*shift)))
        Graphics.write(f"{gamma: .4f}", Graphics.shifted((w//2, h*2//5 - 11*shift)))
        Graphics.write(f"{r0: .4f}", Graphics.shifted((w//2, h*2//5 - 12*shift)))

        tt.update()
        tt.onclick(main)


class Iteration:
    """
    Basic data structure of gathered statistics.

    Methods:
        __init__(self, healthy, infected, recovered): Constructing Day object with number of healthy,
            infected and recovered particles in given iteration.
        __getitem__(self, item): Overridden [] operator.
        infect(self): Adjusting stats to infection event.
        recover(self): Adjusting stats to recovery event.
    """
    def __init__(self, healthy, infected, recovered):
        self.healthy = healthy
        self.infected = infected
        self.recovered = recovered

    def __getitem__(self, item):
        if item == States.INFECTED:
            return self.infected
        if item == States.RECOVERED:
            return self.recovered
        if item == States.HEALTHY:
            return self.healthy
        raise KeyError("Key to Day cell must be a state.")

    def infect(self):
        self.healthy -= 1
        self.infected += 1

    def recover(self):
        self.infected -= 1
        self.recovered += 1


class Data:
    """
    Class made just not to hold list with statistics and information about real-time performing.
    """
    data = [Iteration(GlobalSetup.NO_BALLS - 1, 1, 0)]
    buffering_data = None

    @staticmethod
    def SIR_analyse():
        N = GlobalSetup.NO_BALLS
        S = [el[States.HEALTHY] for el in Data.data]
        I = [el[States.INFECTED] for el in Data.data]
        R = [el[States.RECOVERED] for el in Data.data]

        dSdt = [n - c for c, n in zip(S[:-1], S[1:])]
        dRdt = [n - c for c, n in zip(R[:-1], R[1:])]

        beta_data = [-dSdt[i]*N/I[i]/S[i] for i in range(len(dSdt)) if S[i] > 0]
        beta_estimator = sum(beta_data)/len(beta_data)

        gamma_data = [dRdt[i]/I[i] for i in range(len(dRdt))]
        gamma_estimator = sum(gamma_data)/len(gamma_data)

        R0 = beta_estimator/gamma_estimator
        return beta_estimator, gamma_estimator, R0

    def SIR_analyse_improved(self, precision=.01):
        N = GlobalSetup.NO_BALLS
        S = np.array([el[States.HEALTHY] for el in self.data])
        I = np.array([el[States.INFECTED] for el in self.data])
        R = np.array([el[States.RECOVERED] for el in self.data])
        iterations = len(S)

        loss = {}
        for gamma in np.arange(0, 1, precision):
            for beta in np.arange(0, 1, precision):
                numerical_S, numerical_I = self.model_with_euler_method(gamma, beta, iterations, N)
                difference_S = np.sum((numerical_S - S)**2)
                difference_I = np.sum((numerical_I - I)**2)
                difference = difference_S + difference_I
                loss[(beta, gamma)] = difference

        beta, gamma = max(loss, key=loss.get)
        R0 = beta/gamma

        return beta, gamma, R0

    @staticmethod
    def model_with_euler_method(gamma, beta, iterations, N, precision=100):
        S = [N-1]
        I = [1]
        for _ in iterations*precision:
            ds = -beta * I[-1] * S[-1] / N
            di = beta * I[-1] * S[-1] / N - gamma * I[-1]
            S.append(S[-1] + ds/precision)
            I.append(I[-1] + di/precision)
        return S[::precision], I[::precision]


def main():
    """
    Main loop.
    """
    population = PopulationMethods.constructor()
    PopulationMethods.set_moving(population)
    if Data.buffering_data is None:
        Graphics.init()
        for i in range(GlobalSetup.ITERATIONS):
            Data.data.append(copy.deepcopy(Data.data[-1]))
            Graphics.draw(population)
            PopulationMethods.update_population(population)
            if Data.data[-1][States.INFECTED] == 0:
                break

    else:
        print("Simulating pandemic...")
        for i in range(GlobalSetup.ITERATIONS):
            Data.data.append(copy.deepcopy(Data.data[-1]))
            PopulationMethods.update_population(population)
            Data.buffering_data.append([])
            for ball in population:
                Data.buffering_data[-1].append((ball.center, ball.color))
            if Data.data[-1][States.INFECTED] == 0:
                break
        Graphics.init()
        for states in Data.buffering_data:
            Graphics.offline_draw(states)

    Graphics.draw_stats()
    tt.exitonclick()


if __name__ == "__main__":
    main()
