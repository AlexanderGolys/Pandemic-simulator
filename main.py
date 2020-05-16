import turtle as tt
import random
from enum import Enum, auto
import time
import warnings
import copy


class GlobalSetup:
    ball_radius = 10
    ball_speed = 3000
    FPS = 60
    no_balls = 70               # number of balls
    social_distancing = .2      # not moving balls to all balls ratio
    recovery_time = 500
    window_resolution = (1600, 900)
    iterations = 10000


class Vector:
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
        self.x *= (GlobalSetup.ball_speed/GlobalSetup.FPS) ** (1/2)
        self.y *= (GlobalSetup.ball_speed/GlobalSetup.FPS) ** (1/2)

    def shift(self, coords):
        return coords[0] + self.x, coords[1] + self.y

    def constant_mul(self, c):
        return Vector(self.x*c, self.y*c)


class States(Enum):
    HEALTHY = auto()
    INFECTED = auto()
    RECOVERED = auto()


class Colors:
    GREEN = (0, 0xFF, 0)
    RED = (0xFF, 0, 0)
    YELLOW = (0xFF, 0x6F, 0)
    BG_COLOR = (0xEE, 0xEE, 0xEE)
    WHITE = (0xFF, 0xFF, 0xFF)
    BLACK = (0, 0, 0)

    get_color = {
        States.HEALTHY: GREEN,
        States.INFECTED: RED,
        States.RECOVERED: YELLOW
    }


class Ball:
    def __init__(self, av):
        self.center = list(random.choice(tuple(av)))
        self.x = self.center[0]
        self.y = self.center[1]
        self.radius = GlobalSetup.ball_radius
        self.moving = None
        self.velocity = Vector(random.uniform(-1, 1), random.uniform(-1, 1))
        self.velocity.normalise()
        self.velocity.denormalise_to_speed()
        self.state = States.HEALTHY
        self.color = Colors.get_color[self.state]
        self.infection_time = 0
        self.collision_time = 0

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
        """
        Distance between balls
        """
        x1, y1 = self.center
        x2, y2 = other.center
        return ((x1 - x2)**2 + (y1 - y2)**2)**(1/2) - self.radius - other.radius

    def draw(self):
        tt.up()
        tt.setpos(Graphics.shifted(self.center))
        tt.dot(2*self.radius, self.color)


class PopulationMethods:
    @staticmethod
    def set_moving(population):
        for i, ball in enumerate(population):
            ball.moving = i > GlobalSetup.social_distancing*len(population)

    @staticmethod
    def constructor():
        if GlobalSetup.no_balls > 70:
            warnings.warn("Simulation can be a bit laggy with selected numbers of balls. Suggested number of balls: 70")
        print("Initialising balls...")
        population = []
        available = {(i, j) for i in range(GlobalSetup.ball_radius,
                                           GlobalSetup.window_resolution[0] - GlobalSetup.ball_radius)
                     for j in range(GlobalSetup.ball_radius,
                                    GlobalSetup.window_resolution[1] - GlobalSetup.ball_radius)}
        for _ in range(GlobalSetup.no_balls):
            ball = Ball(available)
            population.append(ball)
            for c1 in range(ball.x - 2*ball.radius, ball.x + 2*ball.radius + 1):
                for c2 in range(ball.y - 2*ball.radius, ball.y + 2*ball.radius + 1):
                    available.difference_update({(c1, c2)})

        population[-1].state = States.INFECTED
        print(f"Initialised {len(population)} samples.")

        return population

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
            data[-1][0] -= 1
            data[-1][1] += 1
        if ball1.state == States.HEALTHY and ball2.state == States.INFECTED:
            ball1.state = States.INFECTED
            data[-1][0] -= 1
            data[-1][1] += 1

    @staticmethod
    def update_population(population):
        for ball in population:
            if ball.moving:
                ball.center = ball.velocity.shift(ball.center)
            if ball.state == States.INFECTED:
                ball.infection_time += 1
                if ball.infection_time > GlobalSetup.recovery_time:
                    ball.state = States.RECOVERED
                    data[-1][1] -= 1
                    data[-1][2] += 1

        ball_set = set(population)
        second_ball_set = set(population)
        for ball1 in ball_set:
            second_ball_set.difference_update({ball1})
            for ball2 in second_ball_set:
                if ball1 - ball2 <= 1:
                    PopulationMethods.balls_collision(ball1, ball2)
                    ball1.collision_time += 1
                    ball2.collision_time += 1
                    if ball1.collision_time > GlobalSetup.FPS/2 and ball2.collision_time > GlobalSetup.FPS/2:
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

            if ball1.x - ball1.radius <= 2:
                ball1.velocity.right()
            if ball1.x + ball1.radius >= GlobalSetup.window_resolution[0]-2:
                ball1.velocity.left()
            if ball1.y - ball1.radius <= 2:
                ball1.velocity.up()
            if ball1.y + ball1.radius >= GlobalSetup.window_resolution[1]-2:
                ball1.velocity.down()


class Graphics:
    @staticmethod
    def init():
        tt.setup(*GlobalSetup.window_resolution)
        tt.mode("logo")
        tt.colormode(255)
        tt.title("Pandemic Simulation")
        tt.tracer(0, 0)
        tt.hideturtle()

    @staticmethod
    def draw(population):
        tt.reset()
        tt.hideturtle()
        for ball in population:
            ball.draw()
        tt.update()
        time.sleep(1/GlobalSetup.FPS)

    @staticmethod
    def shifted(coords):
        x, y = coords
        return x - GlobalSetup.window_resolution[0] // 2, y - GlobalSetup.window_resolution[1] // 2

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
        prev_x_index = 0
        current_x_index = 0
        prev_y = data[current_x_index][what_to_plot] * value_jump + 11 * h // 20
        for current_x_pixel in range(w // 3 + h // 20 + 2, w * 2 // 3 - h // 20):
            current_x_index += jump
            current_y = data[int(current_x_index)][what_to_plot] * value_jump + 11 * h // 20
            Graphics.draw_line(Graphics.shifted((current_x_pixel - 1, prev_y)),
                               Graphics.shifted((current_x_pixel, current_y)), Colors.RED, 2)
            prev_y = current_y

    @staticmethod
    def plot_area(jump, y_axis, w, h):
        current_x_index = 0
        for current_x_pixel in range(w // 3 + h // 20 + 2, w * 2 // 3 - h // 20):
            len1 = y_axis * data[int(current_x_index)][0] / GlobalSetup.no_balls
            len2 = y_axis * data[int(current_x_index)][1] / GlobalSetup.no_balls
            len3 = y_axis * data[int(current_x_index)][2] / GlobalSetup.no_balls
            Graphics.draw_line(Graphics.shifted((current_x_pixel, 11*h//20)),
                               Graphics.shifted((current_x_pixel, 11*h//20+len3)), Colors.YELLOW, 1)
            Graphics.draw_line(Graphics.shifted((current_x_pixel, 11*h//20+len3)),
                               Graphics.shifted((current_x_pixel, 11*h//20+len3+len2)), Colors.RED, 1)
            Graphics.draw_line(Graphics.shifted((current_x_pixel, 11*h//20+len3+len2)),
                               Graphics.shifted((current_x_pixel, 11*h//20+len2+len3+len1)), Colors.GREEN, 1)
            current_x_index += jump

    @staticmethod
    def draw_stats():
        w, h = GlobalSetup.window_resolution

        # draw background
        Graphics.draw_rect(Graphics.shifted((0, 0)), Graphics.shifted((w, h)), Colors.BG_COLOR)
        Graphics.draw_rect(Graphics.shifted((w//3, h//2)), Graphics.shifted((w*2//3, h*9//10)), Colors.WHITE)
        Graphics.draw_rect(Graphics.shifted((w//3, h*2//5)), Graphics.shifted((w*2//3, 0)), Colors.WHITE)

        x_axis_len = w//3 - h//10 - 1
        y_axis_len = h*3//10
        jump = len(data) / x_axis_len
        value_jump = y_axis_len / GlobalSetup.no_balls
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

        # Graphics.plot_line(jump, value_jump, w, h, 0)
        # Graphics.plot_line(jump, value_jump, w, h, 1)
        # Graphics.plot_line(jump, value_jump, w, h, 2)

        tt.update()


def main():
    global data
    data = [[GlobalSetup.no_balls-1, 1, 0]]
    population = PopulationMethods.constructor()
    PopulationMethods.set_moving(population)
    Graphics.init()
    for i in range(GlobalSetup.iterations):
        data.append(copy.deepcopy(data[-1]))
        Graphics.draw(population)
        PopulationMethods.update_population(population)
        if data[-1][1] == 0:
            break
    Graphics.draw_stats()
    tt.exitonclick()


if __name__ == "__main__":
    main()



