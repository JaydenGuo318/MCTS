"""
Author: TA Tony, WU Qihang, HUANG Zhaojun, GUO Junzhe, HU rui
Date: 2323-12-04
Group: 2
Description: an Interactive Program of Gomuku Chess Game
"""

import numpy
import numpy as np
import math
import pygame as pg
import random
import time

# Drawing those dash lines outside the 11x11 board
def draw_dash_line(surface, color, start, end, width=1, dash_length=4):

    x1, y1 = start
    x2, y2 = end
    dl = dash_length

    if (x1 == x2):
        ycoords = [y for y in range(y1, y2, dl if y1 < y2 else -dl)]
        xcoords = [x1] * len(ycoords)
    elif (y1 == y2):
        xcoords = [x for x in range(x1, x2, dl if x1 < x2 else -dl)]
        ycoords = [y1] * len(xcoords)
    else:
        a = abs(x2 - x1)
        b = abs(y2 - y1)
        c = round(math.sqrt(a**2 + b**2))
        dx = dl * a / c
        dy = dl * b / c

        xcoords = [x for x in numpy.arange(x1, x2, dx if x1 < x2 else -dx)]
        ycoords = [y for y in numpy.arange(y1, y2, dy if y1 < y2 else -dy)]

    next_coords = list(zip(xcoords[1::2], ycoords[1::2]))
    last_coords = list(zip(xcoords[0::2], ycoords[0::2]))
    for (x1, y1), (x2, y2) in zip(next_coords, last_coords):
        start = (round(x1), round(y1))
        end = (round(x2), round(y2))
        pg.draw.line(surface, color, start, end, width)


####################################################################################################################
# create the initial empty chess board in the game window
def draw_board():
    
    global xbline, w_size, pad, sep
    
    xbline = bline + 8                        # Add 4 extra line on each boundaries to make chains of 5 that cross boundaries easier to see
    w_size = 720                              # window size
    pad = 36                                  # padding size
    sep = int((w_size-pad*2)/(xbline-1))      # separation between lines = [window size (720) - padding*2 (36*2)]/(Total lines (19) -1)
    
    surface = pg.display.set_mode((w_size, w_size))
    pg.display.set_caption("Gomuku (a.k.a Five-in-a-Row)")
    
    color_line = [0, 0, 0]
    color_board = [241, 196, 15]

    surface.fill(color_board)
    
    for i in range(0, xbline):
        draw_dash_line(surface, color_line, [pad, pad+i*sep], [w_size-pad, pad+i*sep])
        draw_dash_line(surface, color_line, [pad+i*sep, pad], [pad+i*sep, w_size-pad])
        
    for i in range(0, bline):
        pg.draw.line(surface, color_line, [pad+4*sep, pad+(i+4)*sep], [w_size-pad-4*sep, pad+(i+4)*sep], 4)
        pg.draw.line(surface, color_line, [pad+(i+4)*sep, pad+4*sep], [pad+(i+4)*sep, w_size-pad-4*sep], 4)

    pg.display.update()
    
    return surface


####################################################################################################################
# Draw the stones on the board at pos = [row, col]. 
# Draw a black circle at pos if color = 1, and white circle at pos if color =  -1
# row and col are be the indices on the 11x11 board array
# dark gray and light gray circles are also drawn on the dotted grid to indicate a phantom stone piece
def draw_stone(surface, pos, color=0):

    color_black = [0, 0, 0]
    color_dark_gray = [75, 75, 75]
    color_white = [255, 255, 255]
    color_light_gray = [235, 235, 235]
    
    matx = pos[0] + 4 + bline*np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]).flatten()
    matx1 = np.logical_and(matx >= 0, matx < xbline)
    maty = pos[1] + 4 + bline*np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]).T.flatten()
    maty1 = np.logical_and(maty >= 0, maty < xbline)
    mat = np.logical_and(np.logical_and(matx1, maty1), np.array([[True, True, True], [True, False, True], [True, True, True]]).flatten())

    if color==1:
        pg.draw.circle(surface, color_black, [pad+(pos[0]+4)*sep, pad+(pos[1]+4)*sep], 15, 0)
        for f, x, y in zip(mat, matx, maty):
            if f:
                pg.draw.circle(surface, color_dark_gray, [pad+x*sep, pad+y*sep], 15, 0)
                
    elif color==-1:
        pg.draw.circle(surface, color_white, [pad+(pos[0]+4)*sep, pad+(pos[1]+4)*sep], 15, 0)
        for f, x, y in zip(mat, matx, maty):
            if f:
                pg.draw.circle(surface, color_light_gray, [pad+x*sep, pad+y*sep], 15, 0)
        
    pg.display.update()
    

####################################################################################################################
def print_winner(surface, winner=0):
    if winner == 2:
        msg = "Draw! So White wins"
        color = [170,170,170]
    elif winner == 1:
        msg = "Black wins!"
        color = [0,0,0]
    elif winner == -1:
        msg = 'White wins!'
        color = [255,255,255]
    else:
        return
        
    font = pg.font.Font('freesansbold.ttf', 32)
    text = font.render(msg, True, color)
    textRect = text.get_rect()
    textRect.topleft = (0, 0)
    surface.blit(text, textRect)
    pg.display.update()
    
def check_winner(board):
    flag = True
    directions = [(1, 0), (0, 1), (1, 1), (-1, 1)]
    
    for row in range(bline):
        for col in range(bline):
            if board[row][col] == 0:
                flag = False
            else:
                for direction in directions:
                    dx, dy = direction
                    real_x, real_y = row, col
                    cnt = 0
                    while board[real_x][real_y] == board[row][col]:
                        cnt += 1
                        if cnt == 5:
                            return board[row][col]
                        real_x = (real_x + dx) % bline
                        real_y = (real_y + dy) % bline
    if flag == True:
        return 2 # 平局
    else:
        return 0

class Node:
    def __init__(self, state, color, move=None, parent=None, is_to_gameover=0, gameover=0):
        self.state = state                      # 当前棋盘
        self.color = color                      # 结点颜色
        self.move = move                        # 形成当前棋盘的上一动作，元组(x, y)表示行列
        self.parent = parent                    # 父节点
        self.children = []                      # 子节点
        self.win_cnt = 0                        # 模拟得到的胜利数
        self.visit_cnt = 0                      # 访问次数
        self.is_to_gameover = is_to_gameover    # 该节点是否为必胜结点
        self.gameover = gameover                # 该节点已经游戏结束

# 计算uct值
def calc_uct(node, parent_visit_cnt):
    if node.visit_cnt == 0:
        return math.inf # 优先拓展新结点
    else:
        return node.win_cnt / node.visit_cnt + \
            math.sqrt(2 * math.log(parent_visit_cnt) / node.visit_cnt)

# 选择节点
def select_node(node):
    # 选择node下最大uct值的子节点，直至叶子节点
    while node.children:
        best_uct = -1
        best_child = None
        for child in node.children:
            child_uct = calc_uct(child, node.visit_cnt)
            if child_uct > best_uct:
                best_uct = child_uct
                best_child = child
        node = best_child
    return node

# 判断点(row, col)周围是否有棋子（8个方向）
def is_around_chess(board, row, col):
    return board[(row - 1) % bline][(col - 1) % bline] != 0 or \
           board[(row - 1) % bline][col % bline] != 0 or \
           board[(row - 1) % bline][(col + 1) % bline] != 0 or \
           board[row % bline][(col - 1) % bline] != 0 or \
           board[row % bline][(col + 1) % bline] != 0 or \
           board[(row + 1) % bline][(col - 1) % bline] != 0 or \
           board[(row + 1) % bline][col % bline] != 0 or \
           board[(row + 1) % bline][(col + 1) % bline] != 0

# 拓展节点
def expand(node):
    # 仅拓展周围有棋子的位置作为孩子节点
    action_list = []
    for row in range(bline):
        for col in range(bline):
            if node.state[row][col] == 0 and is_around_chess(node.state, row, col):
                action_list.append((row, col))

    # 棋盘已满，不再向下拓展节点，返回空列表
    if check_winner(node.state) != 0 or not action_list:
        return []
    
    for action in action_list:
        expand_one_node(node, action, 0, 0)
        
    return node.children

# 产生一个新节点
def expand_one_node(node, action, is_to_gameover, gameover):
    new_state = [row.copy() for row in node.state]
    new_state[action[0]][action[1]] = node.color
    child_node = Node(state=new_state, color=-node.color, move=(action[0], action[1]), parent=node, is_to_gameover=is_to_gameover, gameover=gameover)
    node.children.append(child_node)

# 在拓展节点前武断认为某个棋局黑或白必胜
def is_to_gameover(node):
    board = node.state
    turn = node.color  # 轮到谁走
    directions = [(1, 0), (0, 1), (1, 1), (-1, 1)]
    
    # 检查是否有一步以内的必胜走法，“活四”->'_1111_', '10111', '11011', '11101'
    one_try_board = [row.copy() for row in board]
    for row in range(bline):
        for col in range(bline):
            if one_try_board[row][col] == 0:
                one_try_board[row][col] = turn
                for direction in directions:
                    dx, dy = direction
                    cnt = 0
                    real_x, real_y = row, col
                    while one_try_board[real_x][real_y] == one_try_board[row][col]:
                        cnt += 1
                        real_x = (real_x + dx) % bline
                        real_y = (real_y + dy) % bline
                    cnt -= 1
                    real_x, real_y = row, col
                    while one_try_board[real_x][real_y] == one_try_board[row][col]:
                        cnt += 1
                        real_x = (real_x - dx) % bline
                        real_y = (real_y - dy) % bline
                    if cnt >= 5:
                        action = (row, col)
                        expand_one_node(node, action, turn, turn)
                        node.is_to_gameover = turn
                        return turn
                one_try_board[row][col] = 0
    
    # 检验对手是否有一步必胜点，若检验没有，则己方可以展开进攻
    one_try_board = [row.copy() for row in board]
    for row in range(bline):
        for col in range(bline):
            if one_try_board[row][col] == 0:
                one_try_board[row][col] = -turn
                for direction in directions:
                    dx, dy = direction
                    cnt = 0
                    real_x, real_y = row, col
                    while one_try_board[real_x][real_y] == one_try_board[row][col]:
                        cnt += 1
                        real_x = (real_x + dx) % bline
                        real_y = (real_y + dy) % bline
                    cnt -= 1
                    real_x, real_y = row, col
                    while one_try_board[real_x][real_y] == one_try_board[row][col]:
                        cnt += 1
                        real_x = (real_x - dx) % bline
                        real_y = (real_y - dy) % bline
                    if cnt >= 5:                        
                        action = (row, col)
                        expand_one_node(node, action, 0, 0)
                        return -turn
                one_try_board[row][col] = 0
    
    # 检查是否有两步以内的必胜走法，“活三”->'_01110_', '010110', '011010'
    one_try_board = [row.copy() for row in board]
    for row in range(bline):
        for col in range(bline):
            if one_try_board[row][col] == 0:
                one_try_board[row][col] = turn
                for direction in directions:
                    dx, dy = direction
                    cnt = 0
                    real_x, real_y = row, col
                    while one_try_board[real_x][real_y] == one_try_board[row][col]:
                        cnt += 1
                        real_x = (real_x + dx) % bline
                        real_y = (real_y + dy) % bline
                    cnt -= 1
                    another_real_x, another_real_y = row, col
                    while one_try_board[another_real_x][another_real_y] == one_try_board[row][col]:
                        cnt += 1
                        another_real_x = (another_real_x - dx) % bline
                        another_real_y = (another_real_y - dy) % bline
                    if cnt == 4 and one_try_board[real_x][real_y] == 0 and \
                            one_try_board[another_real_x][another_real_y] == 0:
                        action = (row, col)
                        expand_one_node(node, action, turn, 0)
                        node.is_to_gameover = turn
                        return turn
                one_try_board[row][col] = 0
                    
    # 检查是否有两步以内的必胜走法，“两个冲四“，“一个冲四一个活三”
    one_try_board = [row.copy() for row in board]
    for row in range(bline):
        for col in range(bline):
            if one_try_board[row][col] == 0:
                one_try_board[row][col] = turn
                for direction in directions:
                    rush_four_one_dir = False
                    for two_try in [1, -1]:
                        dx = two_try * direction[0]
                        dy = two_try * direction[1]
                        cnt = 0
                        real_x, real_y = row, col
                        empty_chance = 1
                        while one_try_board[real_x][real_y] != -turn:
                            if one_try_board[real_x][real_y] != turn:
                                if empty_chance:
                                    empty_chance -= 1
                                else:
                                    break
                            cnt += 1
                            real_x = (real_x + dx) % bline
                            real_y = (real_y + dy) % bline
                        cnt -= 1
                        real_x, real_y = row, col
                        while one_try_board[real_x][real_y] != -turn:
                            if one_try_board[real_x][real_y] != turn:
                                if empty_chance:
                                    empty_chance -= 1
                                else:
                                    break
                            cnt += 1
                            real_x = (real_x - dx) % bline
                            real_y = (real_y - dy) % bline
                        if cnt >= 5:
                            rush_four_one_dir = True
                            break
                    if rush_four_one_dir:
                        directions_copy = directions.copy()
                        directions_copy.remove(direction)
                        for direction in directions_copy:
                            for two_try in [1, -1]:
                                dx = two_try * direction[0]
                                dy = two_try * direction[1]
                                cnt = 0
                                real_x, real_y = row, col
                                empty_chance = 1
                                while one_try_board[real_x][real_y] != -turn:
                                    if one_try_board[real_x][real_y] != turn:
                                        if empty_chance:
                                            empty_chance -= 1
                                        else:
                                            break
                                    cnt += 1
                                    real_x = (real_x + dx) % bline
                                    real_y = (real_y + dy) % bline
                                cnt -= 1
                                real_x, real_y = row, col
                                while one_try_board[real_x][real_y] != -turn:
                                    if one_try_board[real_x][real_y] != turn:
                                        if empty_chance:
                                            empty_chance -= 1
                                        else:
                                            break
                                    cnt += 1
                                    real_x = (real_x - dx) % bline
                                    real_y = (real_y - dy) % bline
                                if cnt >= 5:
                                    action = (row, col)
                                    expand_one_node(node, action, turn, 0)
                                    node.is_to_gameover = turn
                                    return turn
                                if (one_try_board[(row + dx) % bline][(col + dy) % bline] == turn and \
                                    one_try_board[(row + 2 * dx) % bline][(col + 2 * dy) % bline] == turn and \
                                    one_try_board[(row + 3 * dx) % bline][(col + 3 * dy) % bline] == 0 and \
                                    one_try_board[(row - dx) % bline][(col - dy) % bline] == 0 and \
                                    (one_try_board[(row - 2 * dx) % bline][(col - 2 * dy) % bline] != -turn or \
                                     one_try_board[(row + 4 * dx) % bline][(col + 4 * dy) % bline] != -turn)) or \
                                   (one_try_board[(row + dx) % bline][(col + dy) % bline] == turn and \
                                    one_try_board[(row - dx) % bline][(col - dy) % bline] == turn and \
                                    one_try_board[(row + 2 * dx) % bline][(col + 2 * dy) % bline] == 0 and \
                                    one_try_board[(row - 2 * dx) % bline][(col - 2 * dy) % bline] == 0 and \
                                    (one_try_board[(row + 3 * dx) % bline][(col + 3 * dy) % bline] != -turn or \
                                     one_try_board[(row - 3 * dx) % bline][(col - 3 * dy) % bline] != -turn)) or \
                                   (one_try_board[(row - dx) % bline][(col - dy) % bline] == turn and \
                                    one_try_board[(row + dx) % bline][(col + dy) % bline] == 0 and \
                                    one_try_board[(row + 2 * dx) % bline][(col + 2 * dy) % bline] == turn and \
                                    (one_try_board[(row + 3 * dx) % bline][(col + 3 * dy) % bline] != -turn or \
                                     one_try_board[(row - 2 * dx) % bline][(col - 2 * dy) % bline] != -turn)):
                                    action = (row, col)
                                    expand_one_node(node, action, turn, 0)
                                    node.is_to_gameover = turn
                                    return turn
                                
                        break
                        
                one_try_board[row][col] = 0
                
    # 检查对手否是有两步必胜点，若检验没有，则己方可以展开进攻
    one_try_board = [row.copy() for row in board]
    for row in range(bline):
        for col in range(bline):
            if one_try_board[row][col] == 0:
                one_try_board[row][col] = -turn
                for direction in directions:
                    dx, dy = direction
                    cnt = 0
                    real_x, real_y = row, col
                    while one_try_board[real_x][real_y] == one_try_board[row][col]:
                        cnt += 1
                        real_x = (real_x + dx) % bline
                        real_y = (real_y + dy) % bline
                    cnt -= 1
                    another_real_x, another_real_y = row, col
                    while one_try_board[another_real_x][another_real_y] == one_try_board[row][col]:
                        cnt += 1
                        another_real_x = (another_real_x - dx) % bline
                        another_real_y = (another_real_y - dy) % bline
                    if cnt == 4 and one_try_board[real_x][real_y] == 0 and \
                            one_try_board[another_real_x][another_real_y] == 0:
                        action = (row, col)
                        expand_one_node(node, action, 0, 0)
                        return -turn
                one_try_board[row][col] = 0
        
    # 检查是否有三步以内的必胜走法，“两个活三“
    one_try_board = [row.copy() for row in board]
    for row in range(bline):
        for col in range(bline):
            if one_try_board[row][col] == 0:
                loose_three_cnt = 0
                one_try_board[row][col] = turn
                for direction in directions:
                    dx, dy = direction
                    if (one_try_board[(row + dx) % bline][(col + dy) % bline] == turn and \
                     one_try_board[(row - dx) % bline][(col - dy) % bline] == turn and \
                     one_try_board[(row + 2 * dx) % bline][(col + 2 * dy) % bline] == 0 and \
                     one_try_board[(row - 2 * dx) % bline][(col - 2 * dy) % bline] == 0 and \
                     (one_try_board[(row + 3 * dx) % bline][(col + 3 * dy) % bline] != -turn or \
                      one_try_board[(row - 3 * dx) % bline][(col - 3 * dy) % bline] != -turn)):
                        loose_three_cnt += 1
                        if loose_three_cnt == 2:
                            action = (row, col)
                            expand_one_node(node, action, 0, 0)
                            return turn
                    for two_try in [1, -1]:
                        dx = two_try * direction[0]
                        dy = two_try * direction[1]
                        cnt = 0
                        real_x, real_y = row, col
                        if (one_try_board[(row + dx) % bline][(col + dy) % bline] == turn and \
                            one_try_board[(row + 2 * dx) % bline][(col + 2 * dy) % bline] == turn and \
                            one_try_board[(row + 3 * dx) % bline][(col + 3 * dy) % bline] == 0 and \
                            one_try_board[(row - dx) % bline][(col - dy) % bline] == 0 and \
                            (one_try_board[(row - 2 * dx) % bline][(col - 2 * dy) % bline] != -turn or \
                             one_try_board[(row + 4 * dx) % bline][(col + 4 * dy) % bline] != -turn)) or \
                           (one_try_board[(row - dx) % bline][(col - dy) % bline] == turn and \
                            one_try_board[(row + dx) % bline][(col + dy) % bline] == 0 and \
                            one_try_board[(row + 2 * dx) % bline][(col + 2 * dy) % bline] == turn and \
                            (one_try_board[(row + 3 * dx) % bline][(col + 3 * dy) % bline] != -turn or \
                             one_try_board[(row - 2 * dx) % bline][(col - 2 * dy) % bline] != -turn)):
                           loose_three_cnt += 1
                           if loose_three_cnt == 2:
                               action = (row, col)
                               expand_one_node(node, action, 0, 0)
                               return turn
                one_try_board[row][col] = 0
                
    one_try_board = [row.copy() for row in board]
    for row in range(bline):
        for col in range(bline):
            if one_try_board[row][col] == 0:
                loose_three_cnt = 0
                one_try_board[row][col] = -turn
                for direction in directions:
                    dx, dy = direction
                    if (one_try_board[(row + dx) % bline][(col + dy) % bline] == -turn and \
                     one_try_board[(row - dx) % bline][(col - dy) % bline] == -turn and \
                     one_try_board[(row + 2 * dx) % bline][(col + 2 * dy) % bline] == 0 and \
                     one_try_board[(row - 2 * dx) % bline][(col - 2 * dy) % bline] == 0 and \
                     (one_try_board[(row + 3 * dx) % bline][(col + 3 * dy) % bline] != turn or \
                      one_try_board[(row - 3 * dx) % bline][(col - 3 * dy) % bline] != turn)):
                        loose_three_cnt += 1
                        if loose_three_cnt == 2:
                            action = (row, col)
                            expand_one_node(node, action, 0, 0)
                            return -turn
                    for two_try in [1, -1]:
                        dx = two_try * direction[0]
                        dy = two_try * direction[1]
                        cnt = 0
                        real_x, real_y = row, col
                        if (one_try_board[(row + dx) % bline][(col + dy) % bline] == -turn and \
                            one_try_board[(row + 2 * dx) % bline][(col + 2 * dy) % bline] == -turn and \
                            one_try_board[(row + 3 * dx) % bline][(col + 3 * dy) % bline] == 0 and \
                            one_try_board[(row - dx) % bline][(col - dy) % bline] == 0 and \
                            (one_try_board[(row - 2 * dx) % bline][(col - 2 * dy) % bline] != turn or \
                             one_try_board[(row + 4 * dx) % bline][(col + 4 * dy) % bline] != turn)) or \
                           (one_try_board[(row - dx) % bline][(col - dy) % bline] == -turn and \
                            one_try_board[(row + dx) % bline][(col + dy) % bline] == 0 and \
                            one_try_board[(row + 2 * dx) % bline][(col + 2 * dy) % bline] == -turn and \
                            (one_try_board[(row + 3 * dx) % bline][(col + 3 * dy) % bline] != turn or \
                             one_try_board[(row - 2 * dx) % bline][(col - 2 * dy) % bline] != turn)):
                           loose_three_cnt += 1
                           if loose_three_cnt == 2:
                               action = (row, col)
                               expand_one_node(node, action, 0, 0)
                               return -turn
                one_try_board[row][col] = 0
    
    return 0

# 仿真
def simulate(node):
    current_state = [row.copy() for row in node.state]
    action_list = []
    for row in range(bline):
        for col in range(bline):
            if node.state[row][col] == 0 and is_around_chess(node.state, row, col):
                action_list.append((row, col))
    
    turn = node.color
    directions = [(1, 0), (0, 1), (1, 1), (-1, 1)]
    
    while action_list:
        action = random.choice(action_list)
        current_state[action[0]][action[1]] = turn
        
        for direction in directions:
            dx, dy = direction
            cnt = 0
            real_x, real_y = action[0], action[1]
            while current_state[real_x][real_y] == turn:
                cnt += 1
                real_x = (real_x + dx) % bline
                real_y = (real_y + dy) % bline
            cnt -= 1
            real_x, real_y = action[0], action[1]
            while current_state[real_x][real_y] == turn:
                cnt += 1
                real_x = (real_x - dx) % bline
                real_y = (real_y - dy) % bline
            if cnt >= 5:
                return turn
        
        turn = -turn
        for i in range(action[0] - 1, action[0] + 2):
            for j in range(action[1] - 1, action[1] + 2):
                if (i % bline, j % bline) not in action_list:
                    action_list.append((i % bline, j % bline))
        action_list.remove((action[0], action[1]))
    
    return 0

# 反向传播
def backpropagate(node, res):
    while node is not None:
        node.visit_cnt += 1
        if res * node.color == -1:
            node.win_cnt += 1
        node = node.parent

def ai_move(board, ai_color):
    # 主mcts算法
    root = Node(state=board, color=ai_color)
    start = time.time()
    while time.time() - start < 5:
        # 选择一个叶子节点
        leaf = select_node(root)
        
        # 若该叶子节点已经游戏结束，则直接反向传播结果
        if leaf.gameover:
            backpropagate(leaf, leaf.gameover)
        else:
            # 判断该叶子节点是否是必胜棋局，若是必胜棋局则函数中已向下拓展一个节点
            if is_to_gameover(leaf):
                node = leaf.children[0]
                if node.is_to_gameover:
                    backpropagate(node, node.is_to_gameover)
            else:
                new_nodes = expand(leaf)
                if new_nodes:
                    node = random.choice(new_nodes)
                    res = simulate(node)
                else:
                    node = leaf
                    res = check_winner(node.state)
                backpropagate(node, res)

    # 找到根节点的最好孩子
    best_win_rate = -1
    best_child = None
    for node in root.children:
        print(node.move, node.win_cnt, node.visit_cnt, round(node.win_cnt / node.visit_cnt, 3), end='\n')
        if node.visit_cnt != 0:
            win_rate = node.win_cnt / node.visit_cnt
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                best_child = node
    
    if best_child is not None:
        return best_child.move
    else:
        return random.choice(np.argwhere(np.array(board) == 0))
        

def main(player_is_black=True):
    
    global bline
    bline = 11                  # the board size is 11x11 => need to draw 11 lines on the board
    
    pg.init()
    surface = draw_board()
    
    board = np.zeros((bline,bline), dtype=int)

    running = True
    gameover = False
    
    player_move = player_is_black
    ai_color = -1 if player_is_black else 1
    
    while running:
       
        for event in pg.event.get():              # A for loop to process all the events initialized by the player
            
            if event.type == pg.QUIT:             # terminate if player closes the game window 
                running = False
                
            if event.type == pg.MOUSEBUTTONDOWN and not gameover and player_move:        # detect whether the player is clicking in the window
                
                (x,y) = event.pos                                        # check if the clicked position is on the 11x11 center grid
                if (x > pad+3.75*sep) and (x < w_size-pad-3.75*sep) and (y > pad+3.75*sep) and (y < w_size-pad-3.75*sep):
                    col = round((x-pad)/sep-4)     
                    row = round((y-pad)/sep-4)

                    if board[row, col] == 0:                             # update the board matrix if that position has not been occupied
                        color = 1 if player_is_black else -1
                        board[row, col] = color
                        draw_stone(surface, [col, row], color)
                        player_move = not player_move
                        gameover = check_winner(board)
                        print('玩家落子后棋局：')
                        print(board)
                        print('\n\n')

        if not player_move and not gameover:
            row, col = ai_move(board, ai_color)
            print('ai选择的行列是', row, col)
            color = -1 if player_is_black else 1
            board[row, col] = color
            draw_stone(surface, [col, row], color)
            player_move = not player_move
            gameover = check_winner(board)
            print('ai落子后棋局：')
            print(board)
            print('\n\n')

        if gameover:
            print_winner(surface, gameover)
            
    pg.quit()
    
    
if __name__ == '__main__':
    main(True)
