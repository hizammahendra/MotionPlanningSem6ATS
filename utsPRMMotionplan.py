import math
import random
import matplotlib.pyplot as plt
import numpy as np

show_animation = True


class RRT:
    """
    Rapidly-exploring Random Tree (RRT) Path Planning
    """

    class Node:
        """RRT Node"""
        def __init__(self, x: float, y: float):
            self.x = x
            self.y = y
            self.path_x = []
            self.path_y = []
            self.parent = None

    class AreaBounds:
        """Area boundary for play area"""
        def __init__(self, area: list):
            self.xmin = float(area[0])
            self.xmax = float(area[1])
            self.ymin = float(area[2])
            self.ymax = float(area[3])

    def __init__(self,
                 start: list,
                 goal: list,
                 obstacle_x_list: list,
                 obstacle_y_list: list,
                 rand_area: list,
                 expand_dis: float = 3.0,
                 path_resolution: float = 0.5,
                 goal_sample_rate: int = 5,
                 max_iter: int = 2000,
                 play_area: list = None):
        """
        Parameters
        ----------
        start            : [x, y]
        goal             : [x, y]
        obstacle_x_list  : list of obstacle x-coordinates (point obstacles)
        obstacle_y_list  : list of obstacle y-coordinates (point obstacles)
        rand_area        : [min, max]
        play_area        : [xmin, xmax, ymin, ymax] (optional)
        """
        self.start = self.Node(start[0], start[1])
        self.end   = self.Node(goal[0],  goal[1])

        self.min_rand = rand_area[0]
        self.max_rand = rand_area[1]

        self.play_area = self.AreaBounds(play_area) if play_area is not None else None

        self.expand_dis      = expand_dis
        self.path_resolution = path_resolution
        self.goal_sample_rate = goal_sample_rate
        self.max_iter        = max_iter

        # Build obstacle list as (x, y, radius) for collision checking
        # Point obstacles are represented with radius = robot_radius
        self.robot_radius = 3.0  # same as original PRM robot_size
        self.obstacle_list = list(zip(obstacle_x_list, obstacle_y_list,
                                      [self.robot_radius] * len(obstacle_x_list)))

        self.node_list = []

    # ------------------------------------------------------------------
    def planning(self, animation: bool = True):
        """RRT path planning"""
        self.node_list = [self.start]

        for i in range(self.max_iter):
            rnd_node    = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_ind]

            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)

            if (self.check_if_outside_play_area(new_node, self.play_area) and
                    self.check_collision(new_node, self.obstacle_list)):

                self.node_list.append(new_node)

                if animation and i % 5 == 0:
                    self.draw_graph(rnd_node)

                if self.calc_dist_to_goal(new_node.x, new_node.y) <= self.expand_dis:
                    final_node = self.steer(new_node, self.end, self.expand_dis)
                    if self.check_collision(final_node, self.obstacle_list):
                        return self.generate_final_course(len(self.node_list) - 1)

        if animation:
            self.draw_graph()

        print("Cannot find path")
        return None

    # ------------------------------------------------------------------
    def steer(self, from_node, to_node, extend_length=float("inf")):
        new_node = self.Node(from_node.x, from_node.y)
        d, theta = self.calc_distance_and_angle(from_node, to_node)

        new_node.path_x = [new_node.x]
        new_node.path_y = [new_node.y]

        if extend_length > d:
            extend_length = d

        n_expand = math.floor(extend_length / self.path_resolution)

        for _ in range(n_expand):
            new_node.x += self.path_resolution * math.cos(theta)
            new_node.y += self.path_resolution * math.sin(theta)
            new_node.path_x.append(new_node.x)
            new_node.path_y.append(new_node.y)

        d, _ = self.calc_distance_and_angle(new_node, to_node)
        if d <= self.path_resolution:
            new_node.path_x.append(to_node.x)
            new_node.path_y.append(to_node.y)
            new_node.x = to_node.x
            new_node.y = to_node.y

        new_node.parent = from_node
        return new_node

    def generate_final_course(self, goal_ind: int):
        path = [[self.end.x, self.end.y]]
        node = self.node_list[goal_ind]
        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([node.x, node.y])
        return path[::-1]

    def calc_dist_to_goal(self, x, y):
        return math.hypot(x - self.end.x, y - self.end.y)

    def get_random_node(self):
        if random.randint(0, 100) > self.goal_sample_rate:
            return self.Node(
                random.uniform(self.min_rand, self.max_rand),
                random.uniform(self.min_rand, self.max_rand)
            )
        return self.Node(self.end.x, self.end.y)

    # ------------------------------------------------------------------
    def draw_graph(self, rnd=None):
        plt.clf()
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: exit(0) if event.key == 'escape' else None
        )
        if rnd is not None:
            plt.plot(rnd.x, rnd.y, "^k", markersize=6)

        for node in self.node_list:
            if node.parent:
                plt.plot(node.path_x, node.path_y, "-g", linewidth=0.5)

        # Draw point obstacles
        ox = [obs[0] for obs in self.obstacle_list]
        oy = [obs[1] for obs in self.obstacle_list]
        plt.plot(ox, oy, ".k", markersize=2)

        if self.play_area is not None:
            plt.plot([self.play_area.xmin, self.play_area.xmax,
                      self.play_area.xmax, self.play_area.xmin,
                      self.play_area.xmin],
                     [self.play_area.ymin, self.play_area.ymin,
                      self.play_area.ymax, self.play_area.ymax,
                      self.play_area.ymin], "-k")

        plt.plot(self.start.x, self.start.y, "^r", markersize=10)
        plt.plot(self.end.x,   self.end.y,   "^c", markersize=10)
        plt.axis("equal")
        plt.grid(True)
        plt.pause(0.001)

    # ------------------------------------------------------------------
    @staticmethod
    def get_nearest_node_index(node_list, rnd_node):
        distances = [(node.x - rnd_node.x)**2 + (node.y - rnd_node.y)**2
                     for node in node_list]
        return distances.index(min(distances))

    @staticmethod
    def check_if_outside_play_area(node, play_area):
        if play_area is None:
            return True
        return not (node.x < play_area.xmin or node.x > play_area.xmax or
                    node.y < play_area.ymin or node.y > play_area.ymax)

    @staticmethod
    def check_collision(node, obstacle_list):
        if node is None:
            return False
        for (ox, oy, size) in obstacle_list:
            dx_list = [ox - x for x in node.path_x]
            dy_list = [oy - y for y in node.path_y]
            d_list  = [dx*dx + dy*dy for dx, dy in zip(dx_list, dy_list)]
            if min(d_list) <= size**2:
                return False  # collision
        return True

    @staticmethod
    def calc_distance_and_angle(from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        return math.hypot(dx, dy), math.atan2(dy, dx)


# ======================================================================
def main():
    print(__file__ + " start!!")

    # ── Start & goal  ───────────────────────
    sx = 0.0
    sy = 30.0
    gx = 50.0
    gy = 50.0

    # ── Obstacles  ────────
    ox = []
    oy = []

    # Bottom wall
    for i in range(-10, 60):
        ox.append(i)
        oy.append(-10.0)
    # nomor 2
    for i in range(-10, 10):
        ox.append(i)
        oy.append(40.0)

    for i in range(10, 30):
        ox.append(i)
        oy.append(5.0)

    for i in range(40, 60):
        ox.append(i)
        oy.append(5.0)

    for i in range(30, 40):
        ox.append(i)
        oy.append(40.0)

    # Right wall
    for i in range(-10, 60):
        ox.append(60.0)
        oy.append(i)
    # Top wall
    for i in range(-10, 61):
        ox.append(i)
        oy.append(60.0)
    # Left wall
    for i in range(-10, 61):
        ox.append(-10.0)
        oy.append(i)
    # Internal obstacles
    for i in range(10, 40):
        ox.append(i)
        oy.append(20)

    for i in range(15):
        ox.append(30.0)
        oy.append(20.0 - i)

    for i in range(20):
        ox.append(30.0)
        oy.append(60.0 - i)

    for i in range(20):
        ox.append(10.0)
        oy.append(40.0 - i)

    for i in range(0):
        ox.append(i)
        oy.append(10.0)
    # ───────────────────────────────────────────────────────────────────

    if show_animation:
        plt.plot(ox, oy, ".k")
        plt.plot(sx, sy, "^r")
        plt.plot(gx, gy, "^c")
        plt.grid(True)
        plt.axis("equal")

    rrt = RRT(
        start=[sx, sy],
        goal=[gx, gy],
        obstacle_x_list=ox,
        obstacle_y_list=oy,
        rand_area=[-10, 60],
        expand_dis=3.0,
        path_resolution=0.5,
        goal_sample_rate=5,
        max_iter=5000,
    )

    path = rrt.planning(animation=show_animation)

    if path is None:
        print("Cannot find path")
    else:
        print("Path found!")
        if show_animation:
            rrt.draw_graph()
            plt.plot([p[0] for p in path], [p[1] for p in path], '-r', linewidth=2)
            plt.grid(True)
            plt.pause(0.001)
            plt.show()


if __name__ == '__main__':
    main()