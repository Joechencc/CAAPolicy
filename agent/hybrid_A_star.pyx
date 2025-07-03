"""
modified from  Hybrid A* @author: Huiming Zhou
"""
import sys
import math
import heapq
from heapdict import heapdict
import numpy as np
from scipy.spatial import cKDTree 

cimport cython
cimport numpy as np
from cython cimport dict 

from libc.time cimport clock, CLOCKS_PER_SEC

##### some parameters that can be changed
cdef float BEV_range_x = 32
cdef float BEV_range_y = 32 

cdef float MAX_LENGTH = 40 #200.0 ## meter?

# Parameter config
cdef float PI = math.pi
cdef float XY_RESO = 2.0  # [m]
cdef float YAW_RESO = np.deg2rad(15.0)  # [rad] ## previous is 15
cdef float MOVE_STEP = 0.2  # [m] path interporate resolution
cdef float N_STEER = 20.0  # steer command number
cdef int COLLISION_CHECK_STEP = 1  # skip number for collision check


cdef float GEAR_COST = 100.0  # switch back penalty cost
cdef float BACKWARD_COST = 5.0  # backward penalty cost
cdef float STEER_CHANGE_COST = 2.0 # 5.0 # steer angle change penalty cost
cdef float STEER_ANGLE_COST = 1.0  # steer angle penalty cost
cdef float H_COST = 10 #5.0  # Heuristic cost penalty cost

cdef float RF = 3.8   # [m] distance from rear to vehicle front end of vehicle
cdef float RB = 1.0   # [m] distance from rear to vehicle back end of vehicle
cdef float W = 2.2    # [m] width of vehicle
cdef float WD = 0.7 * W  # [m] distance between left-right wheels
cdef float WB = 2.9   # [m] Wheel base
cdef float TR = 0.4   # [m] Tyre radius
cdef float TW = 0.8     # [m] Tyre width
cdef float MAX_STEER = 0.6  # [rad] maximum steering angle ## try to limit to -35 to 35, then carla takes -0.5 to 0.5
cdef float BUBBLE_R = np.hypot((RF + RB) / 2.0, W / 2.0)+0.05  # bubble radius
#########################################################################
##### import HybridAstarPlanner.astar as astar
cdef class Node_astar:
    cdef float cost 
    cdef int x, y, pind
    def __init__(self, x: int, y: int, cost: float, pind: int):
        self.x = x  # x position of node
        self.y = y  # y position of node
        self.cost = cost  # g cost of node
        self.pind = pind  # parent index of node


cdef class Para_astar:
    cdef:
        int minx, miny, maxx, maxy, xw, yw
        float reso
        list[list] motion  
    def __init__(self, minx: int, miny: int, maxx: int, maxy: int, xw: int, yw: int, reso: float, motion: list[list]):
        self.minx = minx
        self.miny = miny
        self.maxx = maxx
        self.maxy = maxy
        self.xw = xw
        self.yw = yw
        self.reso = reso  # resolution of grid world
        self.motion = motion  # motion set

cdef float u_cost(list[int] u):
    return math.hypot(u[0], u[1])

cdef int calc_index_astar(Node_astar node, Para_astar P):
    return (node.y - P.miny) * P.xw + (node.x - P.minx)

cdef tuple[Para_astar, list[list[bint]]] calc_parameters_astar(list ox, list oy, float reso, float rr): ##rr: robot radius
    cdef int minx, miny, maxx, maxy, xw, yw
    cdef list[list[int]] motion 
    cdef Para_astar P 
    cdef list[list[bint]] obsmap
    minx, miny = round(min(ox)), round(min(oy))
    maxx, maxy = round(max(ox)), round(max(oy))
    xw, yw = round(BEV_range_x), round(BEV_range_y) #maxx - minx, maxy - miny

    motion = [[-1, 0], [-1, 1], [0, 1], [1, 1],[1, 0], [1, -1], [0, -1], [-1, -1]] #get_motion()
    P = Para_astar(minx, miny, maxx, maxy, xw, yw, reso, motion)
    obsmap = calc_obsmap(ox, oy, rr, P)

    return P, obsmap

cdef list[list[bint]] calc_obsmap(list ox, list oy, float rr, Para_astar P):
    cdef list[list[bint]] obsmap = [[False for _ in range(P.yw)] for _ in range(P.xw)]
    cdef int x, xx, y, yy 
    cdef float oxx, oyy 

    for x in range(P.xw):
        xx = x + P.minx
        for y in range(P.yw):
            yy = y + P.miny
            for oxx, oyy in zip(ox, oy):
                if math.hypot(oxx - xx, oyy - yy) <= rr / P.reso:
                    obsmap[x][y] = True
                    break
    return obsmap    

cdef bint check_node_astar(Node_astar node, Para_astar P, list[list[bint]] obsmap):
    if node.x <= P.minx or node.x >= P.maxx or node.y <= P.miny or node.y >= P.maxy or node.x - P.minx >= P.xw or node.y - P.miny >= P.yw:
        return False

    if obsmap[node.x - P.minx][node.y - P.miny]:
        return False
    return True

cdef list[list] calc_holonomic_heuristic_with_obstacle(Node node, list ox, list oy, float reso, float rr):
    cdef Node_astar n_goal, n_curr, n, new_node, tmp_astar_node 
    cdef float x, y, node_cost 
    cdef Para_astar P
    cdef list[list[bint]] obsmap
    cdef dict open_set, closed_set 
    cdef list q_priority
    cdef int ind, goal_ind, i, n_ind  
    cdef bint check 
    cdef list[list] hmap

    n_goal = Node_astar(round(node.x[-1] / reso), round(node.y[-1] / reso), 0.0, -1)

    ox = [x / reso for x in ox]
    oy = [y / reso for y in oy]

    P, obsmap = calc_parameters_astar(ox, oy, reso, rr)

    open_set, closed_set = dict(), dict()

    goal_ind = calc_index_astar(n_goal, P)
    open_set[goal_ind] = n_goal

    q_priority = []
    heapq.heappush(q_priority, (n_goal.cost, goal_ind))

    while True:
        if not open_set:
            break

        _, ind = heapq.heappop(q_priority)
        n_curr = open_set[ind]
        closed_set[ind] = n_curr
        open_set.pop(ind)

        for i in range(len(P.motion)):
            new_node = Node_astar(n_curr.x + P.motion[i][0],
                        n_curr.y + P.motion[i][1],
                        n_curr.cost + u_cost(P.motion[i]), ind)

            
            node_cost = new_node.cost
            check = check_node_astar(new_node, P, obsmap)
            if not check:
                continue

            n_ind = calc_index_astar(new_node, P)
            if n_ind not in closed_set:
                if n_ind in open_set:
                    tmp_astar_node = open_set[n_ind]
                    
                    if tmp_astar_node.cost > node_cost:
                        tmp_astar_node.cost = node_cost
                        tmp_astar_node.pind = ind
                        open_set[n_ind] = tmp_astar_node
                else:
                    open_set[n_ind] = new_node
                    heapq.heappush(q_priority, (node_cost, n_ind))

    hmap = [[np.inf for _ in range(P.yw)] for _ in range(P.xw)]

    #print('hmap shape: ', P.xw, P.yw, P.minx, P.miny)
    for n in closed_set.values():
        #print('n.x - P.minx', n.x - P.minx, ' n.y - P.miny: ', n.y - P.miny)
        hmap[n.x - P.minx][n.y - P.miny] = n.cost

    return hmap

#########################################################################
##### import CurvesGenerator.reeds_shepp as rs


# class for PATH element
cdef class PATH_rs:
    cdef list lengths, x, y, yaw
    cdef list[str] ctypes
    cdef list[int] directions
    cdef float L   
    def __init__(self, lengths: list, ctypes: list[str], L: float, x: list, y: list, yaw: list, directions: list[int]):
        self.lengths = lengths              # lengths of each part of path (+: forward, -: backward) [float]
        self.ctypes = ctypes                # type of each part of the path [string]
        self.L = L                          # total path length [float]
        self.x = x                          # final x positions [m]
        self.y = y                          # final y positions [m]
        self.yaw = yaw                      # final yaw angles [rad]
        self.directions = directions        # forward: 1, backward:-1

cdef float pi_2_pi(float theta):
    while theta > math.pi:
        theta -= 2.0 * math.pi

    while theta < -math.pi:
        theta += 2.0 * math.pi

    return theta

cdef tuple[list, list, list, list[int]] interpolate_rs(int ind, float l, str m, float maxc, float ox, float oy, float oyaw, list px, list py, list pyaw, list[int] directions):
    cdef float ldy, gdx, gdy 
    if m == "S":
        px[ind] = ox + l / maxc * math.cos(oyaw)
        py[ind] = oy + l / maxc * math.sin(oyaw)
        pyaw[ind] = oyaw
    else:
        ldx = math.sin(l) / maxc
        if m == "WB":
            ldy = (1.0 - math.cos(l)) / maxc
        elif m == "R":
            ldy = (1.0 - math.cos(l)) / (-maxc)

        gdx = math.cos(-oyaw) * ldx + math.sin(-oyaw) * ldy
        gdy = -math.sin(-oyaw) * ldx + math.cos(-oyaw) * ldy
        px[ind] = ox + gdx
        py[ind] = oy + gdy

    if m == "WB":
        pyaw[ind] = oyaw + l
    elif m == "R":
        pyaw[ind] = oyaw - l

    if l > 0.0:
        directions[ind] = 1
    else:
        directions[ind] = -1

    return px, py, pyaw, directions

cdef list[PATH_rs] set_path_rs(list paths, list lengths, list ctypes):
    cdef PATH_rs path, path_e
    cdef float i  

    path = PATH_rs([], [], 0.0, [], [], [], [])
    path.ctypes = ctypes
    path.lengths = lengths

    # check same path exist
    for path_e in paths:
        if path_e.ctypes == path.ctypes:
            if sum([x - y for x, y in zip(path_e.lengths, path.lengths)]) <= 0.01:
                return paths  # not insert path

    path.L = sum([abs(i) for i in lengths])

    if path.L >= MAX_LENGTH:
        return paths

    assert path.L >= 0.01
    paths.append(path)

    return paths

cdef tuple[float, float] R(float x, float y):
    """
    Return the polar coordinates (r, theta) of the point (x, y)
    """
    cdef float r, theta 
    r = math.hypot(x, y)
    theta = math.atan2(y, x)

    return r, theta


cdef float M(float theta):
    """
    Regulate theta to -pi <= theta < pi
    """
    cdef float phi 

    phi = theta % (2.0 * math.pi)

    if phi < -math.pi:
        phi += 2.0 * math.pi
    if phi > math.pi:
        phi -= 2.0 * math.pi

    return phi



cdef tuple[bint, float, float, float] LSL(float x, float y, float phi):
    cdef float u, t 
    u, t = R(x - math.sin(phi), y - 1.0 + math.cos(phi))

    if t >= 0.0:
        v = M(phi - t)
        if v >= 0.0:
            return True, t, u, v

    return False, 0.0, 0.0, 0.0


cdef tuple[bint, float, float, float] LSR(float x, float y, float phi):
    cdef float u1, t1, u, theta, t, v   
    u1, t1 = R(x + math.sin(phi), y - 1.0 - math.cos(phi))
    u1 = u1 ** 2

    if u1 >= 4.0:
        u = math.sqrt(u1 - 4.0)
        theta = math.atan2(2.0, u)
        t = M(t1 + theta)
        v = M(t - phi)

        if t >= 0.0 and v >= 0.0:
            return True, t, u, v

    return False, 0.0, 0.0, 0.0


cdef tuple[bint, float, float, float] LRL(float x, float y, float phi):
    cdef float u1, t1, u, t, v 
    u1, t1 = R(x - math.sin(phi), y - 1.0 + math.cos(phi))

    if u1 <= 4.0:
        u = -2.0 * math.asin(0.25 * u1)
        t = M(t1 + 0.5 * u + math.pi)
        v = M(phi - t + u)

        if t >= 0.0 and u <= 0.0:
            return True, t, u, v

    return False, 0.0, 0.0, 0.0


cdef list[PATH_rs] SCS(float x, float y, float phi, list paths):
    cdef bint flag
    cdef float t, u, v
    flag, t, u, v = SLS(x, y, phi)

    if flag:
        paths = set_path_rs(paths, [t, u, v], ["S", "WB", "S"])

    flag, t, u, v = SLS(x, -y, -phi)
    if flag:
        paths = set_path_rs(paths, [t, u, v], ["S", "R", "S"])

    return paths


cdef tuple[bint, float, float, float] SLS(float x, float y, float phi):
    cdef float xd, t, u, v 
    phi = M(phi)

    if y > 0.0 and 0.0 < phi < math.pi * 0.99:
        xd = -y / math.tan(phi) + x
        t = xd - math.tan(phi / 2.0)
        u = phi
        v = math.sqrt((x - xd) ** 2 + y ** 2) - math.tan(phi / 2.0)
        return True, t, u, v
    elif y < 0.0 and 0.0 < phi < math.pi * 0.99:
        xd = -y / math.tan(phi) + x
        t = xd - math.tan(phi / 2.0)
        u = phi
        v = -math.sqrt((x - xd) ** 2 + y ** 2) - math.tan(phi / 2.0)
        return True, t, u, v

    return False, 0.0, 0.0, 0.0


cdef list[PATH_rs] CSC(float x, float y, float phi, list paths):
    cdef bint flag
    cdef float t, u, v
    flag, t, u, v = LSL(x, y, phi)
    if flag:
        paths = set_path_rs(paths, [t, u, v], ["WB", "S", "WB"])

    flag, t, u, v = LSL(-x, y, -phi)
    if flag:
        paths = set_path_rs(paths, [-t, -u, -v], ["WB", "S", "WB"])

    flag, t, u, v = LSL(x, -y, -phi)
    if flag:
        paths = set_path_rs(paths, [t, u, v], ["R", "S", "R"])

    flag, t, u, v = LSL(-x, -y, phi)
    if flag:
        paths = set_path_rs(paths, [-t, -u, -v], ["R", "S", "R"])

    flag, t, u, v = LSR(x, y, phi)
    if flag:
        paths = set_path_rs(paths, [t, u, v], ["WB", "S", "R"])

    flag, t, u, v = LSR(-x, y, -phi)
    if flag:
        paths = set_path_rs(paths, [-t, -u, -v], ["WB", "S", "R"])

    flag, t, u, v = LSR(x, -y, -phi)
    if flag:
        paths = set_path_rs(paths, [t, u, v], ["R", "S", "WB"])

    flag, t, u, v = LSR(-x, -y, phi)
    if flag:
        paths = set_path_rs(paths, [-t, -u, -v], ["R", "S", "WB"])

    return paths


cdef list[PATH_rs] CCC(float x, float y, float phi, list paths):
    cdef bint flag
    cdef float t, u, v, xb, yb 


    flag, t, u, v = LRL(x, y, phi)
    if flag:
        paths = set_path_rs(paths, [t, u, v], ["WB", "R", "WB"])

    flag, t, u, v = LRL(-x, y, -phi)
    if flag:
        paths = set_path_rs(paths, [-t, -u, -v], ["WB", "R", "WB"])

    flag, t, u, v = LRL(x, -y, -phi)
    if flag:
        paths = set_path_rs(paths, [t, u, v], ["R", "WB", "R"])

    flag, t, u, v = LRL(-x, -y, phi)
    if flag:
        paths = set_path_rs(paths, [-t, -u, -v], ["R", "WB", "R"])

    # backwards
    xb = x * math.cos(phi) + y * math.sin(phi)
    yb = x * math.sin(phi) - y * math.cos(phi)

    flag, t, u, v = LRL(xb, yb, phi)
    if flag:
        paths = set_path_rs(paths, [v, u, t], ["WB", "R", "WB"])

    flag, t, u, v = LRL(-xb, yb, -phi)
    if flag:
        paths = set_path_rs(paths, [-v, -u, -t], ["WB", "R", "WB"])

    flag, t, u, v = LRL(xb, -yb, -phi)
    if flag:
        paths = set_path_rs(paths, [v, u, t], ["R", "WB", "R"])

    flag, t, u, v = LRL(-xb, -yb, phi)
    if flag:
        paths = set_path_rs(paths, [-v, -u, -t], ["R", "WB", "R"])

    return paths


cdef tuple[float, float] calc_tauOmega(float u, float v, float xi, float eta, float phi):
    cdef float delta, A, B, t1, t2, tau, omega

    delta = M(u - v)
    A = math.sin(u) - math.sin(delta)
    B = math.cos(u) - math.cos(delta) - 1.0

    t1 = math.atan2(eta * A - xi * B, xi * A + eta * B)
    t2 = 2.0 * (math.cos(delta) - math.cos(v) - math.cos(u)) + 3.0

    if t2 < 0:
        tau = M(t1 + math.pi)
    else:
        tau = M(t1)

    omega = M(tau - u + v - phi)

    return tau, omega


cdef tuple[bint, float, float, float] LRLRn(float x, float y, float phi):
    cdef float xi, eta, rho, u, t, v 

    xi = x + math.sin(phi)
    eta = y - 1.0 - math.cos(phi)
    rho = 0.25 * (2.0 + math.sqrt(xi * xi + eta * eta))

    if rho <= 1.0:
        u = math.acos(rho)
        t, v = calc_tauOmega(u, -u, xi, eta, phi)
        if t >= 0.0 and v <= 0.0:
            return True, t, u, v

    return False, 0.0, 0.0, 0.0


cdef tuple[bint, float, float, float] LRLRp(float x, float y, float phi):
    cdef float xi, eta, rho, u, t, v

    xi = x + math.sin(phi)
    eta = y - 1.0 - math.cos(phi)
    rho = (20.0 - xi * xi - eta * eta) / 16.0

    if 0.0 <= rho <= 1.0:
        u = -math.acos(rho)
        if u >= -0.5 * math.pi:
            t, v = calc_tauOmega(u, u, xi, eta, phi)
            if t >= 0.0 and v >= 0.0:
                return True, t, u, v

    return False, 0.0, 0.0, 0.0


cdef list[PATH_rs] CCCC(float x, float y, float phi, list paths):
    cdef bint flag
    cdef float t, u, v 


    flag, t, u, v = LRLRn(x, y, phi)
    if flag:
        paths = set_path_rs(paths, [t, u, -u, v], ["WB", "R", "WB", "R"])

    flag, t, u, v = LRLRn(-x, y, -phi)
    if flag:
        paths = set_path_rs(paths, [-t, -u, u, -v], ["WB", "R", "WB", "R"])

    flag, t, u, v = LRLRn(x, -y, -phi)
    if flag:
        paths = set_path_rs(paths, [t, u, -u, v], ["R", "WB", "R", "WB"])

    flag, t, u, v = LRLRn(-x, -y, phi)
    if flag:
        paths = set_path_rs(paths, [-t, -u, u, -v], ["R", "WB", "R", "WB"])

    flag, t, u, v = LRLRp(x, y, phi)
    if flag:
        paths = set_path_rs(paths, [t, u, u, v], ["WB", "R", "WB", "R"])

    flag, t, u, v = LRLRp(-x, y, -phi)
    if flag:
        paths = set_path_rs(paths, [-t, -u, -u, -v], ["WB", "R", "WB", "R"])

    flag, t, u, v = LRLRp(x, -y, -phi)
    if flag:
        paths = set_path_rs(paths, [t, u, u, v], ["R", "WB", "R", "WB"])

    flag, t, u, v = LRLRp(-x, -y, phi)
    if flag:
        paths = set_path_rs(paths, [-t, -u, -u, -v], ["R", "WB", "R", "WB"])

    return paths


cdef tuple[bint, float, float, float] LRSR(float x, float y, float phi):
    cdef float xi, eta, rho, theta, t, u, v  

    xi = x + math.sin(phi)
    eta = y - 1.0 - math.cos(phi)
    rho, theta = R(-eta, xi)

    if rho >= 2.0:
        t = theta
        u = 2.0 - rho
        v = M(t + 0.5 * math.pi - phi)
        if t >= 0.0 and u <= 0.0 and v <= 0.0:
            return True, t, u, v

    return False, 0.0, 0.0, 0.0


cdef tuple[bint, float, float, float] LRSL(float x, float y, float phi):
    cdef float xi, eta, rho, theta, r, u, t, v 

    xi = x - math.sin(phi)
    eta = y - 1.0 + math.cos(phi)
    rho, theta = R(xi, eta)

    if rho >= 2.0:
        r = math.sqrt(rho * rho - 4.0)
        u = 2.0 - r
        t = M(theta + math.atan2(r, -2.0))
        v = M(phi - 0.5 * math.pi - t)
        if t >= 0.0 and u <= 0.0 and v <= 0.0:
            return True, t, u, v

    return False, 0.0, 0.0, 0.0


cdef list[PATH_rs] CCSC(float x, float y, float phi, list paths):
    cdef bint flag
    cdef float t, u, v, xb, yb  
     

    flag, t, u, v = LRSL(x, y, phi)
    if flag:
        paths = set_path_rs(paths, [t, -0.5 * math.pi, u, v], ["WB", "R", "S", "WB"])

    flag, t, u, v = LRSL(-x, y, -phi)
    if flag:
        paths = set_path_rs(paths, [-t, 0.5 * math.pi, -u, -v], ["WB", "R", "S", "WB"])

    flag, t, u, v = LRSL(x, -y, -phi)
    if flag:
        paths = set_path_rs(paths, [t, -0.5 * math.pi, u, v], ["R", "WB", "S", "R"])

    flag, t, u, v = LRSL(-x, -y, phi)
    if flag:
        paths = set_path_rs(paths, [-t, 0.5 * math.pi, -u, -v], ["R", "WB", "S", "R"])

    flag, t, u, v = LRSR(x, y, phi)
    if flag:
        paths = set_path_rs(paths, [t, -0.5 * math.pi, u, v], ["WB", "R", "S", "R"])

    flag, t, u, v = LRSR(-x, y, -phi)
    if flag:
        paths = set_path_rs(paths, [-t, 0.5 * math.pi, -u, -v], ["WB", "R", "S", "R"])

    flag, t, u, v = LRSR(x, -y, -phi)
    if flag:
        paths = set_path_rs(paths, [t, -0.5 * math.pi, u, v], ["R", "WB", "S", "WB"])

    flag, t, u, v = LRSR(-x, -y, phi)
    if flag:
        paths = set_path_rs(paths, [-t, 0.5 * math.pi, -u, -v], ["R", "WB", "S", "WB"])

    # backwards
    xb = x * math.cos(phi) + y * math.sin(phi)
    yb = x * math.sin(phi) - y * math.cos(phi)

    flag, t, u, v = LRSL(xb, yb, phi)
    if flag:
        paths = set_path_rs(paths, [v, u, -0.5 * math.pi, t], ["WB", "S", "R", "WB"])

    flag, t, u, v = LRSL(-xb, yb, -phi)
    if flag:
        paths = set_path_rs(paths, [-v, -u, 0.5 * math.pi, -t], ["WB", "S", "R", "WB"])

    flag, t, u, v = LRSL(xb, -yb, -phi)
    if flag:
        paths = set_path_rs(paths, [v, u, -0.5 * math.pi, t], ["R", "S", "WB", "R"])

    flag, t, u, v = LRSL(-xb, -yb, phi)
    if flag:
        paths = set_path_rs(paths, [-v, -u, 0.5 * math.pi, -t], ["R", "S", "WB", "R"])

    flag, t, u, v = LRSR(xb, yb, phi)
    if flag:
        paths = set_path_rs(paths, [v, u, -0.5 * math.pi, t], ["R", "S", "R", "WB"])

    flag, t, u, v = LRSR(-xb, yb, -phi)
    if flag:
        paths = set_path_rs(paths, [-v, -u, 0.5 * math.pi, -t], ["R", "S", "R", "WB"])

    flag, t, u, v = LRSR(xb, -yb, -phi)
    if flag:
        paths = set_path_rs(paths, [v, u, -0.5 * math.pi, t], ["WB", "S", "WB", "R"])

    flag, t, u, v = LRSR(-xb, -yb, phi)
    if flag:
        paths = set_path_rs(paths, [-v, -u, 0.5 * math.pi, -t], ["WB", "S", "WB", "R"])

    return paths


cdef tuple[bint, float, float, float] LRSLR(float x, float y, float phi):
    # formula 8.11 *** TYPO IN PAPER ***
    cdef float xi, eta, rho, theta, u, t, v 
    xi = x + math.sin(phi)
    eta = y - 1.0 - math.cos(phi)
    rho, theta = R(xi, eta)

    if rho >= 2.0:
        u = 4.0 - math.sqrt(rho * rho - 4.0)
        if u <= 0.0:
            t = M(math.atan2((4.0 - u) * xi - 2.0 * eta, -2.0 * xi + (u - 4.0) * eta))
            v = M(t - phi)

            if t >= 0.0 and v >= 0.0:
                return True, t, u, v

    return False, 0.0, 0.0, 0.0


cdef list[PATH_rs] CCSCC(float x, float y, float phi, list paths):
    cdef bint flag
    cdef float t, u, v  
  

    flag, t, u, v = LRSLR(x, y, phi)
    if flag:
        paths = set_path_rs(paths, [t, -0.5 * math.pi, u, -0.5 * math.pi, v], ["WB", "R", "S", "WB", "R"])

    flag, t, u, v = LRSLR(-x, y, -phi)
    if flag:
        paths = set_path_rs(paths, [-t, 0.5 * math.pi, -u, 0.5 * math.pi, -v], ["WB", "R", "S", "WB", "R"])

    flag, t, u, v = LRSLR(x, -y, -phi)
    if flag:
        paths = set_path_rs(paths, [t, -0.5 * math.pi, u, -0.5 * math.pi, v], ["R", "WB", "S", "R", "WB"])

    flag, t, u, v = LRSLR(-x, -y, phi)
    if flag:
        paths = set_path_rs(paths, [-t, 0.5 * math.pi, -u, 0.5 * math.pi, -v], ["R", "WB", "S", "R", "WB"])

    return paths




cdef list[PATH_rs] generate_path_rs(list q0, list q1, float maxc):
    cdef float dx, dy, dth, c, s, x, y 
    cdef list[PATH_rs] paths 

    dx = q1[0] - q0[0]
    dy = q1[1] - q0[1]
    dth = q1[2] - q0[2]
    c = math.cos(q0[2])
    s = math.sin(q0[2])
    x = (c * dx + s * dy) * maxc
    y = (-s * dx + c * dy) * maxc

    paths = []
    paths = SCS(x, y, dth, paths)
    paths = CSC(x, y, dth, paths)
    paths = CCC(x, y, dth, paths)
    paths = CCCC(x, y, dth, paths)
    paths = CCSC(x, y, dth, paths)
    paths = CCSCC(x, y, dth, paths)

    return paths   

cdef tuple[list, list, list, list[int]] generate_local_course(float L, list lengths, list[str] mode, float maxc, float step_size):
    cdef int point_num, ind, i 
    cdef list px, py, pyaw, 
    cdef list[int] directions
    cdef float d, ll, l, ox, oy, oyaw, pd 
    cdef str m   

    point_num = int(L / step_size) + len(lengths) + 3

    px = [0.0 for _ in range(point_num)]
    py = [0.0 for _ in range(point_num)]
    pyaw = [0.0 for _ in range(point_num)]
    directions = [0 for _ in range(point_num)]
    ind = 1

    if lengths[0] > 0.0:
        directions[0] = 1
    else:
        directions[0] = -1

    if lengths[0] > 0.0:
        d = step_size
    else:
        d = -step_size

    ll = 0.0

    for m, l, i in zip(mode, lengths, range(len(mode))):
        if l > 0.0:
            d = step_size
        else:
            d = -step_size

        ox, oy, oyaw = px[ind], py[ind], pyaw[ind]

        ind -= 1
        if i >= 1 and (lengths[i - 1] * lengths[i]) > 0:
            pd = -d - ll
        else:
            pd = d - ll

        while abs(pd) <= abs(l):
            ind += 1
            px, py, pyaw, directions = \
                interpolate_rs(ind, pd, m, maxc, ox, oy, oyaw, px, py, pyaw, directions)
            pd += d

        ll = l - pd - d  # calc remain length

        ind += 1
        px, py, pyaw, directions = \
            interpolate_rs(ind, l, m, maxc, ox, oy, oyaw, px, py, pyaw, directions)

    if len(px) <= 1:
        return [], [], [], []

    # remove unused data
    while len(px) >= 1 and px[-1] == 0.0:
        px.pop()
        py.pop()
        pyaw.pop()
        directions.pop()

    return px, py, pyaw, directions 

cdef list[PATH_rs] calc_all_paths(float sx, float sy, float syaw, float gx, float gy, float gyaw, float maxc, float step_size=0.2):
    cdef list q0, q1 
    cdef list[PATH_rs] paths
    cdef PATH_rs path 
    cdef list x, y, yaw
    cdef list[int] directions  
    cdef float ix, iy, iyaw, l 
    
    q0 = [sx, sy, syaw]
    q1 = [gx, gy, gyaw]

    paths = generate_path_rs(q0, q1, maxc)

    for path in paths:
        x, y, yaw, directions = generate_local_course(path.L, path.lengths, path.ctypes, maxc, step_size * maxc)

        # convert global coordinate
        path.x = [math.cos(-q0[2]) * ix + math.sin(-q0[2]) * iy + q0[0] for (ix, iy) in zip(x, y)]
        path.y = [-math.sin(-q0[2]) * ix + math.cos(-q0[2]) * iy + q0[1] for (ix, iy) in zip(x, y)]
        path.yaw = [pi_2_pi(iyaw + q0[2]) for iyaw in yaw]
        path.directions = directions
        path.lengths = [l / maxc for l in path.lengths]
        path.L = path.L / maxc

    return paths


#########################################################################


cdef class Node:
    cdef int xind, yind, yawind, direction, pind 
    cdef float steer, cost 
    cdef list directions, x, y, yaw  
    def __init__(self, xind: int, yind: int, yawind: int, direction: int, x: list, y: list,
                 yaw: list, directions:list, steer: float, cost: float, pind: int):
        self.xind = xind
        self.yind = yind
        self.yawind = yawind
        self.direction = direction
        self.x = x
        self.y = y
        self.yaw = yaw
        self.directions = directions
        self.steer = steer
        self.cost = cost
        self.pind = pind


cdef class Para:
    cdef int minx, miny, minyaw, maxx, maxy, maxyaw, xw, yw, yaww
    cdef float xyreso, yawreso 
    cdef list ox, oy 
    cdef object kdtree
    def __init__(self, minx: int, miny: int, minyaw: int, maxx: int, maxy: int, maxyaw: int,
                 xw: int, yw: int, yaww: int, xyreso: float, yawreso: float, ox: list, oy: list, kdtree: object):
        self.minx = minx
        self.miny = miny
        self.minyaw = minyaw
        self.maxx = maxx
        self.maxy = maxy
        self.maxyaw = maxyaw
        self.xw = xw
        self.yw = yw
        self.yaww = yaww
        self.xyreso = xyreso
        self.yawreso = yawreso
        self.ox = ox
        self.oy = oy
        self.kdtree = kdtree


cdef class Path:
    cdef list x, y, yaw, 
    cdef float cost 
    cdef list[int] direction 
    def __init__(self, x: list, y: list, yaw: list, direction: list[int], cost: float):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.direction = direction
        self.cost = cost


cdef class QueuePrior:
    cdef object queue 
    cdef int item 
    cdef float priority
    def __init__(self):
        self.queue = heapdict()

    def empty(self):
        return len(self.queue) == 0  # if Q is empty

    def put(self, item, priority):
        self.queue[item] = priority  # push 

    def get(self):
        return self.queue.popitem()[0]  # pop out element with smallest priority




cdef Path extract_path(dict closed, Node ngoal, Node nstart):
    cdef list rx, ry, ryaw, direc
    cdef float cost 
    cdef Node node 
    cdef bint same_n
    cdef Path path 

    rx, ry, ryaw, direc = [], [], [], []
    cost = 0.0
    node = ngoal

    while True:
        rx += node.x[::-1]
        ry += node.y[::-1]
        ryaw += node.yaw[::-1]
        direc += node.directions[::-1]
        cost += node.cost

        same_n = is_same_grid(node, nstart)
        if same_n:
            break

        node = closed[node.pind]

    rx = rx[::-1]
    ry = ry[::-1]
    ryaw = ryaw[::-1]
    direc = direc[::-1]

    direc[0] = direc[1]
    path = Path(rx, ry, ryaw, direc, cost)

    return path


cdef Node calc_next_node(Node n_curr, int c_id, float u, float d, Para P):
    cdef float step, cost 
    cdef int nlist, i, xind, yind, yawind, direction  
    cdef list xlist, ylist, yawlist
    cdef list[int] directions
    cdef bint is_index_o
    cdef Node node 

    step = XY_RESO * 1.5 ## 2

    nlist = math.ceil(step / MOVE_STEP)
    xlist = [n_curr.x[-1] + d * MOVE_STEP * math.cos(n_curr.yaw[-1])]
    ylist = [n_curr.y[-1] + d * MOVE_STEP * math.sin(n_curr.yaw[-1])]
    yawlist = [pi_2_pi(n_curr.yaw[-1] + d * MOVE_STEP / WB * math.tan(u))]

    for i in range(nlist - 1):
        xlist.append(xlist[i] + d * MOVE_STEP * math.cos(yawlist[i]))
        ylist.append(ylist[i] + d * MOVE_STEP * math.sin(yawlist[i]))
        yawlist.append(pi_2_pi(yawlist[i] + d * MOVE_STEP / WB * math.tan(u)))

    xind = round(xlist[-1] / P.xyreso)
    yind = round(ylist[-1] / P.xyreso)
    yawind = round(yawlist[-1] / P.yawreso)

    is_index_o = is_index_ok(xind, yind, xlist, ylist, yawlist, P)
    if not is_index_o:
        return None

    cost = 0.0

    if d > 0:
        direction = 1
        cost += abs(step)
    else:
        direction = -1
        cost += abs(step) * BACKWARD_COST

    if direction != n_curr.direction:  # switch back penalty
        cost += GEAR_COST

    cost += STEER_ANGLE_COST * abs(u)  # steer angle penalyty
    cost += STEER_CHANGE_COST * abs(n_curr.steer - u)  # steer change penalty
    cost = n_curr.cost + cost

    directions = [direction for _ in range(len(xlist))]

    node = Node(xind, yind, yawind, direction, xlist, ylist,
                yawlist, directions, u, cost, c_id)

    return node


cdef bint is_index_ok(int xind, int yind, list xlist, list ylist, list yawlist, Para P):
    cdef int  k 
    cdef list nodex, nodey, nodeyaw, ind  
    cdef bint collision_or

    if xind <= P.minx or \
            xind >= P.maxx or \
            yind <= P.miny or \
            yind >= P.maxy:
        return False

    ind = list(range(0, len(xlist), COLLISION_CHECK_STEP))

    nodex = [xlist[k] for k in ind]
    nodey = [ylist[k] for k in ind]
    nodeyaw = [yawlist[k] for k in ind]

    collision_or = is_collision(nodex, nodey, nodeyaw, P)
    if collision_or:
        return False

    return True


cdef tuple[bint, Node] update_node_with_analystic_expantion(Node n_curr, Node ngoal, Para P):
    cdef object path 
    cdef list fx, fy, fyaw  
    cdef list[int] fd
    cdef float fcost, fsteer 
    cdef int fpind 
    cdef Node fpath 
    cdef PATH_rs tmp_rs_path

    path = analystic_expantion(n_curr, ngoal, P)  # rs path: n -> ngoal

    if not path:
        return False, None

    tmp_rs_path = path
    fx = tmp_rs_path.x[1:-1]
    fy = tmp_rs_path.y[1:-1]
    fyaw = tmp_rs_path.yaw[1:-1]
    fd = tmp_rs_path.directions[1:-1]

    fcost = n_curr.cost + calc_rs_path_cost(tmp_rs_path)
    fpind = calc_index(n_curr, P)
    fsteer = 0.0

    fpath = Node(n_curr.xind, n_curr.yind, n_curr.yawind, n_curr.direction,
                 fx, fy, fyaw, fd, fsteer, fcost, fpind)

    return True, fpath


cdef object analystic_expantion(Node node, Node ngoal, Para P):
    cdef float sx, sy, syaw, gx, gy, gyaw, maxc, cost_val
    cdef list[PATH_rs] paths 
    cdef QueuePrior pq
    cdef PATH_rs path 
    cdef list ind, pathx, pathy, pathyaw  
    cdef int k 
    cdef bint collision_or  

    sx, sy, syaw = node.x[-1], node.y[-1], node.yaw[-1]
    gx, gy, gyaw = ngoal.x[-1], ngoal.y[-1], ngoal.yaw[-1]

    maxc = math.tan(MAX_STEER) / WB
    paths = calc_all_paths(sx, sy, syaw, gx, gy, gyaw, maxc, step_size=MOVE_STEP)

    if not paths:
        return None

    pq = QueuePrior()
    for path in paths:
        cost_val = calc_rs_path_cost(path)
        pq.put(path, cost_val)

    while not pq.empty():
        path = pq.get()
        ind = list(range(0, len(path.x), COLLISION_CHECK_STEP))

        pathx = [path.x[k] for k in ind]
        pathy = [path.y[k] for k in ind]
        pathyaw = [path.yaw[k] for k in ind]

        collision_or = is_collision(pathx, pathy, pathyaw, P)
        if not collision_or:
            return path

    return None


cdef bint is_collision(list x, list y, list yaw, Para P):
    cdef float d, dl, r, ix, iy, iyaw, cx, cy, xo, yo, dx, dy    
    cdef list ids 
    cdef int i  

    d = 0.05 ### safety range
    dl = (RF - RB) / 2.0 ### rear center to center of the car
    r = (RF + RB) / 2.0 + d

    for ix, iy, iyaw in zip(x, y, yaw):
        

        cx = ix + dl * math.cos(iyaw)
        cy = iy + dl * math.sin(iyaw)

        ids = P.kdtree.query_ball_point([cx, cy], r)

        if not ids:
            continue

        for i in ids:
            xo = P.ox[i] - cx
            yo = P.oy[i] - cy
            dx = xo * math.cos(iyaw) + yo * math.sin(iyaw)
            dy = -xo * math.sin(iyaw) + yo * math.cos(iyaw)

            if abs(dx) < r and abs(dy) < W / 2 + d:
                return True

    return False


cdef float calc_rs_path_cost(PATH_rs rspath):
    cdef float cost, lr 
    cdef int i, nctypes 
    cdef str ctype 
    cdef list ulist 

    cost = 0.0
    for lr in rspath.lengths:
        if lr >= 0:
            cost += 1
        else:
            cost += abs(lr) * BACKWARD_COST

    for i in range(len(rspath.lengths) - 1):
        if rspath.lengths[i] * rspath.lengths[i + 1] < 0.0:
            cost += GEAR_COST

    for ctype in rspath.ctypes:
        if ctype != "S":
            cost += STEER_ANGLE_COST * abs(MAX_STEER)

    nctypes = len(rspath.ctypes)
    ulist = [0.0 for _ in range(nctypes)]

    for i in range(nctypes):
        if rspath.ctypes[i] == "R":
            ulist[i] = -MAX_STEER
        elif rspath.ctypes[i] == "WB":
            ulist[i] = MAX_STEER

    for i in range(nctypes - 1):
        cost += STEER_CHANGE_COST * abs(ulist[i + 1] - ulist[i])

    return cost


cdef float calc_hybrid_cost(Node node, list[list] hmap, Para P):
    cdef float cost 

    cost = node.cost + H_COST * hmap[node.xind - P.minx][node.yind - P.miny]

    return cost


cdef tuple[list, list] calc_motion_set():
    cdef np.ndarray[np.float64_t, ndim=1] s 
    cdef list steer, direc  

    s = np.arange(MAX_STEER / N_STEER,
                  MAX_STEER, MAX_STEER / N_STEER)

    steer = list(s) + [0.0] + list(-s)
    direc = [1.0 for _ in range(len(steer))] + [-1.0 for _ in range(len(steer))]
    steer = steer + steer

    return steer, direc


cdef bint is_same_grid(Node node1, Node node2):
    if node1.xind != node2.xind or \
            node1.yind != node2.yind or \
            node1.yawind != node2.yawind:
        return False

    return True


cdef int calc_index(Node node, Para P):
    cdef int ind 
    ind = (node.yawind - P.minyaw) * P.xw * P.yw + \
          (node.yind - P.miny) * P.xw + \
          (node.xind - P.minx)

    return ind


cdef Para calc_parameters(list ox, list oy, float xyreso, float yawreso, object kdtree):
    cdef int minx, miny, maxx, maxy, xw, yw, minyaw, maxyaw, yaww      
    cdef Para para_sample 
    if len(ox) == 0: ### no obstacles
        ox = [0]
        oy = [0]
    minx = round(min(ox) / xyreso)
    miny = round(min(oy) / xyreso)
    maxx = round(BEV_range_x) #round(max(ox) / xyreso)
    maxy = round(BEV_range_y) #round(max(oy) / xyreso)

    xw, yw = round(BEV_range_x), round(BEV_range_y) #maxx - minx, maxy - miny

    minyaw = round(-PI / yawreso) - 1
    maxyaw = round(PI / yawreso)
    yaww = maxyaw - minyaw

    para_sample = Para(minx, miny, minyaw, maxx, maxy, maxyaw,
                xw, yw, yaww, xyreso, yawreso, ox, oy, kdtree)
    return para_sample


cpdef list[list] hybrid_astar_planning(float sx, float sy, float syaw, float gx, float gy, float gyaw, list ox, list oy):
    cdef Path final_path
    cdef int sxr, syr, gxr, gyr, syawr, gyawr
    cdef Node nstart, ngoal, n_curr, fpath, fnode, node, tmp_node  
    cdef object kdtree
    cdef Para P
    cdef list[list] hmap
    cdef list steer_set, direc_set  
    cdef dict open_set, closed_set 
    cdef int start_ind, ind, node_ind  
    cdef float start_cost, current_cost, start_time, end_time 
    cdef bint update, flag

    sxr, syr = round(sx / XY_RESO), round(sy / XY_RESO)
    gxr, gyr = round(gx / XY_RESO), round(gy / XY_RESO)
    syawr = round(pi_2_pi(syaw) / YAW_RESO)
    gyawr = round(pi_2_pi(gyaw) / YAW_RESO)

    nstart = Node(sxr, syr, syawr, 1, [sx], [sy], [syaw], [1], 0.0, 0.0, -1)
    ngoal = Node(gxr, gyr, gyawr, 1, [gx], [gy], [gyaw], [1], 0.0, 0.0, -1)

    kdtree = cKDTree(np.vstack((ox, oy)).T) 
    P = calc_parameters(ox, oy, XY_RESO, YAW_RESO, kdtree)

    hmap = calc_holonomic_heuristic_with_obstacle(ngoal, P.ox, P.oy, P.xyreso, BUBBLE_R)
    steer_set, direc_set = calc_motion_set()
    start_ind = calc_index(nstart, P)
    open_set, closed_set = {start_ind: nstart}, {}

    start_cost = calc_hybrid_cost(nstart, hmap, P)
    qp = QueuePrior()
    qp.put(start_ind, start_cost)

    flag = False
    start_time = clock()
    while True:
        end_time = clock()
        if (end_time - start_time) / CLOCKS_PER_SEC > 2:
            print('Exceding the time limit 2 seconds. Return None Path')
            flag = True
            break

        if not open_set:
            return None

        ind = qp.get()
        n_curr = open_set[ind]
        closed_set[ind] = n_curr
        open_set.pop(ind)

        update, fpath = update_node_with_analystic_expantion(n_curr, ngoal, P)

        if update:
            fnode = fpath
            break

        for i in range(len(steer_set)):
            node = calc_next_node(n_curr, ind, steer_set[i], direc_set[i], P)

            if not node:
                continue

            node_ind = calc_index(node, P)

            if node_ind in closed_set:
                continue

            current_cost = calc_hybrid_cost(node, hmap, P)
            if node_ind not in open_set:
                open_set[node_ind] = node
                qp.put(node_ind, current_cost)
            else:
                tmp_node = open_set[node_ind]
                if tmp_node.cost > node.cost:
                    open_set[node_ind] = node
                    qp.put(node_ind, current_cost)

    if flag:
        return None 
    else:    
        final_path = extract_path(closed_set, fnode, nstart)
        return [final_path.x, final_path.y, final_path.yaw, final_path.direction]


cpdef tuple[list, list] obstacle_process(np.ndarray[np.uint8_t, ndim=2] matrix, int flag):  
    cdef list ox = []
    cdef list oy = []
    cdef float cell_size = 0.1 
    cdef int x, y 
    cdef float x_a, y_a
    cdef int rows = len(matrix)
    cdef int columns = len(matrix[0])
    for y in range(rows):
        for x in range(columns):
            if matrix[y][x] == flag:
                for x_a, y_a in [[0,0], [cell_size, 0], [cell_size, cell_size], [0, cell_size]]: 
                    ox.append(x*cell_size+x_a)
                    oy.append((rows-y)*cell_size+y_a)
    return (ox, oy)                