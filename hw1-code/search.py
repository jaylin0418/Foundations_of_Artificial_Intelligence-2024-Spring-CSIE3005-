# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Michael Abir (abir2@illinois.edu) on 08/28/2018
# Modified by Shang-Tse Chen (stchen@csie.ntu.edu.tw) on 03/03/2022

"""
This is the main entry point for HW1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# Search should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi,fast)

from queue import PriorityQueue
from copy import deepcopy

def search(maze, searchMethod):
    return {
        "bfs": bfs,
        "astar": astar,
        "astar_corner": astar_corner,
        "astar_multi": astar_multi,
        "fast": fast,
    }.get(searchMethod)(maze)

def backtrace(parent, start, node):
    subPath = []
    backPtr = node
    while backPtr != start:
        subPath.append(backPtr)
        backPtr = parent[backPtr]
        subPath.reverse()
    return subPath

def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    
    path = []
    parent = {}
    explored = set()
    fringe = []
        
    path.append(maze.getStart())
    fringe.append(maze.getStart())
    objective = maze.getObjectives()
    totalObjective = len(objective) # Total dots
    nowReached = 0 # Dots reached
    nowStart = maze.getStart()
    
    while len(fringe) != 0:
        
        node = fringe[0]
        fringe.pop(0)
        if node in explored:
            continue

        if node in objective:
            nowReached += 1
            #find subpath part
            subPath = []
            backPtr = node
            while backPtr != nowStart:
                subPath.append(backPtr)
                backPtr = parent[backPtr]
            subPath.reverse()
            path += subPath
            
            if nowReached == totalObjective:
                break
            
            #starting to find the next dot
            parent = {}
            explored = set()
            fringe = [node]
            nowStart = node
            if node in objective:
                objective.remove(node)
       
        explored.add(node)
        neighbors = maze.getNeighbors(node[0], node[1])
        for neighbor in neighbors:
            if neighbor not in explored:
                fringe.append(neighbor)
                parent[neighbor] = node
    #print(path)    
    return path

def nearesrManhattan(objective, current):
    #this is the heuristic function for A* search
    #("currentPos:",current)
    minDistance = float('inf')
    for obj in objective:
        distance = abs(obj[0] - current[0]) + abs(obj[1] - current[1])
        if distance < minDistance:
            minDistance = distance
    return minDistance
    

def astar(maze):
    #oneNode A*
    path = []
    parent = {}
    explored = set()
    fringe = PriorityQueue()
        
    path.append(maze.getStart())
    parent[maze.getStart()] = None
    #fringe.put((0, maze.getStart()))
    objective = maze.getObjectives()
    totalObjective = len(objective) # Total dots
    nowReached = 0 # Dots reached
    nowStart = maze.getStart()
    fringe.put((nearesrManhattan(objective, nowStart), maze.getStart()))
    cost = {}
    
    cost[nowStart] = 0
    
    while fringe.empty() == False:
        
        first = fringe.get()
        node = first[1] 
        #print(node)
        #print(first[0])
        
        if node in explored:
            continue

        if node in objective:
            nowReached += 1
            subPath = []
            backPtr = node
            while backPtr != nowStart:
                subPath.append(backPtr)
                backPtr = parent[backPtr]
            subPath.reverse()
            path += subPath
            if nowReached == totalObjective:
                break
            parent = {}
            explored = set()
            fringe = PriorityQueue()
            fringe.put((len(path), node))
            nowStart = node
            if node in objective:
                objective.remove(node)
                
        explored.add(node)
        neighbors = maze.getNeighbors(node[0], node[1])
        for neighbor in neighbors:
            if neighbor not in explored:
                newCost = cost[node] + 1
                #print(node, newCost)
                if (not neighbor in cost) or newCost < cost[neighbor]:
                    cost[neighbor] = newCost
                    priority = nearesrManhattan(objective, neighbor) + newCost
                    #print("g: ", len(backtrace(parent, nowStart, node)) + len(path), "h: ", nearesrManhattan(objective, neighbor), "f: ", priority)
                    fringe.put((priority, neighbor))
                    parent[neighbor] = node         
                
    return path
     

class state:
    def __init__(self, row, col, fx,gx, unvisitedNodes):
        self.position = (row, col)
        self.f = fx
        self.g = gx
        self.unvisited = deepcopy(unvisitedNodes)
    def __lt__(self, other):
        if self.f == other.f:
            return self.g < other.g
        return self.f < other.f
  
def astar_corner(maze):
    """
    Runs A star for part 2 of the assignment in the case where there are four corner objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    return astar_multi(maze)

def getManhattanDistance(node1, node2):
    return abs(node1[0] - node2[0]) + abs(node1[1] - node2[1])

#for this function, we need to calculate the minimum spanning tree weight
#edge is the dixtionary that contains the distance between two nodes
#objective is the list of objectives(remaining dots)

def getMstWeight(edges, vertices, nowPos):
    if not vertices or not edges:
        return 0

    # Initialize variables
    mst_weight = 0
    visited = set()
    visited.add(vertices[0])

    # Main loop to build MST
    while len(visited) < len(vertices):
        min_weight = float('inf')
        min_edge = None

        # Find the minimum weight edge to an unvisited vertex
        for v in visited:
            for neighbor in vertices:
                if neighbor not in visited and ((v, neighbor) in edges or (neighbor, v) in edges):
                    if (v, neighbor) in edges:
                        weight = edges[(v, neighbor)]
                    else:
                        weight = edges[(neighbor, v)]
                    if weight < min_weight:
                        min_weight = weight
                        min_edge = (v, neighbor)

        # If no valid edge is found, the graph is not connected
        if min_edge is None:
            return float('inf')  # Or some other appropriate handling

        # Update MST weight and visited set
        mst_weight += min_weight
        visited.add(min_edge[1])

    # Calculate the Manhattan distance to the nearest unvisited vertex from the current position
    nearest_manhattan = min((abs(nowPos[0] - v[0]) + abs(nowPos[1] - v[1])) for v in vertices)
    #print(mst_weight,nearest_manhattan)
    #print("mst_weight: ",mst_weight)
    # Combine MST weight with the minimum Manhattan distance
    return mst_weight + nearest_manhattan

#This function will be called whenever we find our A* solution
def backtrace_AStarmulti(parent, position):
    currentPos = position
    currentObj = []  
    path = []
    path.append(currentPos)
    
    while parent[(currentPos, tuple(currentObj))] != None:
        #print("prev: ",parent[(currentPos, tuple(currentObj))])
        currentPos, currentObj = parent[(currentPos, tuple(currentObj))]
        path.insert(0, currentPos)
        
    return path

def astar_multi(maze):
    startPos = maze.getStart()
    objective = maze.getObjectives()
    fringe = PriorityQueue()
    path = []
    edge = {}
    
    '''參考github adityavgupta的寫法'''
    for i in range(len(objective)):
        for j in range(i+1, len(objective)):
            #edge[(objective[i],objective[j])] = getManhattanDistance(objective[i], objective[j])
            c_maze = deepcopy(maze)
            c_maze.setStart(objective[i])
            c_maze.setObjectives([objective[j]])
            dist = len(astar(c_maze))
            edge[(objective[i],objective[j])] = dist-1

    
    
    parent = {} #parent要用(現在的位置, [沒吃的dots])當key
    visited = set()
    mst = {}
    
    costs = {}
    costs[(startPos, tuple(objective))] = 0
    
    parent[(startPos, tuple(objective))] = None
    start = state(startPos[0], startPos[1], getMstWeight(edge, tuple(objective) ,startPos) ,0, deepcopy(objective))
    fringe.put(start)
    #
    # print("start: ",startPos, start.f)
    
    while fringe.empty() == False:
        current = fringe.get()
        currentPos = current.position
        #print(currentPos, current.f, current.g)
        if len(current.unvisited) == 0:
            path = backtrace_AStarmulti(parent, currentPos)
            #print(edge)
            #("edge: ",edge)
            return path
        
        #look at the neighbors
        neighbors = maze.getNeighbors(currentPos[0], currentPos[1])
        
        for neighbor in neighbors:
            #如果今天neighbor是objective的情況
            if neighbor in current.unvisited:
                newUnvisited = deepcopy(current.unvisited)
                newUnvisited.remove(neighbor)
                #nowAndUnvisited = (neighbor, tuple(newUnvisited))
            else:
                newUnvisited = deepcopy(current.unvisited)
            #如果今天找到更好的cost或是cost根本不存在，就更新
            if ((neighbor, tuple(newUnvisited)) not in costs or costs[(neighbor, tuple(newUnvisited))] > costs[(currentPos, tuple(current.unvisited))]+1):
                costs[(neighbor, tuple(newUnvisited))] = costs[(currentPos, tuple(current.unvisited))]+1
                nowCost = costs[(neighbor, tuple(newUnvisited))]
                parent[(neighbor, tuple(newUnvisited))] = (currentPos, tuple(deepcopy(current.unvisited)))
                #new = state(neighbor[0], neighbor[1], getMstWeight[(tuple(newUnvisited) ,neighbor)] + nearesrManhattan(newUnvisited, neighbor), costs[(neighbor, tuple(newUnvisited))], newUnvisited)
                prevf = current.f
                newf = getMstWeight(edge, newUnvisited, neighbor) + nowCost
                if(newf <= prevf):
                    new = state(neighbor[0], neighbor[1], prevf, nowCost, newUnvisited)
                    fringe.put(new)
                else:
                    new = state(neighbor[0], neighbor[1], newf, nowCost, newUnvisited)
                    fringe.put(new)
    
    #return []


def fast(maze):
    """
    Runs suboptimal search algorithm for part 4.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    #TODO: Write your code here
    return bfs(maze)
    return []

