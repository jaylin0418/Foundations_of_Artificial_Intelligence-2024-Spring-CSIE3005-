# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        #print(successorGameState)
        newPos = successorGameState.getPacmanPosition()
        #print(newPos)
        newFood = successorGameState.getFood()
        #print(newFood)
        newGhostStates = successorGameState.getGhostStates()
        #print(newGhostStates)
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        #print(newScaredTimes)

        "*** YOUR CODE HERE ***"
        #和ghost的曼哈頓距離越遠越好
        distToAllGhost = []
        for ghost in newGhostStates:
            distToAllGhost.append(manhattanDistance(newPos, ghost.getPosition()))
        #print(distToAllGhost)
        minGhost = min(distToAllGhost)
        minGhostIndex = distToAllGhost.index(minGhost)
        
        #和food的曼哈頓距離越靠近越好
        distToAllFoods = []
        for food in currentGameState.getFood().asList():
            distToAllFoods.append(manhattanDistance(newPos, food))
        #print(distToAllFoods)
        minDots = min(distToAllFoods)
        #get the index of the nearest Foods
        minFoodIndex = distToAllFoods.index(minDots)

        #The higher, the better
        return (minGhost + min(newScaredTimes) * 0.5) / (minDots + 0.1)
    
def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        return self.maximizer(gameState, 0, 1)
        
        
        
    def maximizer(self, nowState, index, depth):
        
        if nowState.isWin() or nowState.isLose() or depth == -1:
            return self.evaluationFunction(nowState)
        
        maxValue = float('-inf')
        
        possibleActions = nowState.getLegalActions(index)
        actionToTake = None
        for action in possibleActions:
            newState = nowState.generateSuccessor(index, action)
            tmp = self.minimizer(newState, index + 1, depth + 1)
            #maxValue = max(maxValue, tmp)
            if maxValue < tmp:
                maxValue = tmp 
                actionToTake = action
        
        if depth == 1:
            if actionToTake == None:
                return Directions.STOP
            return actionToTake
        else: 
            return maxValue
        
    def minimizer(self, nowState, index, depth):
        
        if nowState.isWin() or nowState.isLose() or depth == -1:
            return self.evaluationFunction(nowState)
        
        possibleActions = nowState.getLegalActions(index)
        minValue = float('inf')
        
        #consider the case if we reach the last ghost with the last depth
        if(index == nowState.getNumAgents() - 1 and depth == (self.depth) * nowState.getNumAgents()):
            for action in possibleActions:
                minValue = min(minValue, self.evaluationFunction(nowState.generateSuccessor(index, action)))
            
        #consider the case if we reach the last ghost but not the last depth:
        elif index == nowState.getNumAgents() - 1:
            for action in possibleActions:
                minValue = min(minValue, self.maximizer(nowState.generateSuccessor(index, action), 0, depth + 1))
        #considert the case if we are not the last ghost
        else:
            for action in possibleActions:
                minValue = min(minValue, self.minimizer(nowState.generateSuccessor(index, action), index + 1, depth + 1))
        return minValue
    


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.maximizer(gameState, 0, 1, float('-inf'), float('inf'))
        
    def maximizer(self, nowState, index, depth, alpha, beta):
        
        if nowState.isWin() or nowState.isLose() or depth == -1:
            return self.evaluationFunction(nowState)
        
        maxValue = float('-inf')
        
        possibleActions = nowState.getLegalActions(index)
        actionToTake = None
        for action in possibleActions:
            newState = nowState.generateSuccessor(index, action)
            tmp = self.minimizer(newState, index + 1, depth + 1, alpha, beta)
            #maxValue = max(maxValue, tmp)
            if maxValue < tmp:
                maxValue = tmp 
                actionToTake = action
            if maxValue > beta:
                return maxValue
            alpha = max(alpha, maxValue)
        
        if depth == 1:
            if actionToTake == None:
                return Directions.STOP
            return actionToTake
        else: 
            return maxValue
        
    def minimizer(self, nowState, index, depth, alpha, beta):
        
        if nowState.isWin() or nowState.isLose() or depth == -1:
            return self.evaluationFunction(nowState)
        
        possibleActions = nowState.getLegalActions(index)
        minValue = float('inf')
        
        #consider the case if we reach the last ghost with the last depth
        if(index == nowState.getNumAgents() - 1 and depth == (self.depth) * nowState.getNumAgents()):
            for action in possibleActions:
                minValue = min(minValue, self.evaluationFunction(nowState.generateSuccessor(index, action)))
                if minValue < alpha:
                    return minValue
                beta = min(beta, minValue)
            
        #consider the case if we reach the last ghost but not the last depth:
        elif index == nowState.getNumAgents() - 1:
            for action in possibleActions:
                minValue = min(minValue, self.maximizer(nowState.generateSuccessor(index, action), 0, depth + 1, alpha, beta))
                if minValue < alpha:
                    return minValue
                beta = min(beta, minValue)
        #considert the case if we are not the last ghost
        else:
            for action in possibleActions:
                minValue = min(minValue, self.minimizer(nowState.generateSuccessor(index, action), index + 1, depth + 1 , alpha, beta))
                if minValue < alpha:
                    return minValue
                beta = min(beta, minValue)
           
        
        '''
        if minValue <= alpha:
                return minValue
            beta = min(beta, minValue) 
        '''
        return minValue
        
    

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.maximizer(gameState, 0, 1)
        #util.raiseNotDefined()
        
    def maximizer(self, nowState, index, depth):
        
        if nowState.isWin() or nowState.isLose() or depth == -1:
            return self.evaluationFunction(nowState)
        
        maxValue = float('-inf')
        
        possibleActions = nowState.getLegalActions(index)
        actionToTake = None
        for action in possibleActions:
            newState = nowState.generateSuccessor(index, action)
            tmp = self.minimizer(newState, index + 1, depth + 1)
            #maxValue = max(maxValue, tmp)
            if maxValue < tmp:
                maxValue = tmp 
                actionToTake = action
        
        if depth == 1:
            if actionToTake == None:
                return Directions.STOP
            return actionToTake
        else: 
            return maxValue
    
    def minimizer(self, nowState, index, depth):
        
        if nowState.isWin() or nowState.isLose() or depth == -1:
            return self.evaluationFunction(nowState)
        
        possibleActions = nowState.getLegalActions(index)
        minValue = float('inf')
        
        values = []
        #consider the case if we reach the last ghost with the last depth
        if(index == nowState.getNumAgents() - 1 and depth == (self.depth) * nowState.getNumAgents()):
            for action in possibleActions:
                values.append(self.evaluationFunction(nowState.generateSuccessor(index, action)))
                #minValue = min(minValue, self.evaluationFunction(nowState.generateSuccessor(index, action)))
            
        #consider the case if we reach the last ghost but not the last depth:
        elif index == nowState.getNumAgents() - 1:
            for action in possibleActions:
                values.append(self.maximizer(nowState.generateSuccessor(index, action), 0, depth + 1))
                #minValue = min(minValue, self.maximizer(nowState.generateSuccessor(index, action), 0, depth + 1))
        #considert the case if we are not the last ghost
        else:
            for action in possibleActions:
                values.append(self.minimizer(nowState.generateSuccessor(index, action), index + 1, depth + 1))
                #minValue = min(minValue, self.minimizer(nowState.generateSuccessor(index, action), index + 1, depth + 1))
        return sum(values) / len(values)   
    
    
    

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    The formula I use: current game state score / (sum of the distance to all the ghosts + 1 + the minimum distance to all the food) + stateScore
    """
    "*** YOUR CODE HERE ***"
    
    pacmanPos = currentGameState.getPacmanPosition()
    foodMatrix = currentGameState.getFood().asList()
    stateScore = 0
    #calculate the distance to all the ghosts
    
    distToAllGhost = []
    for ghost in currentGameState.getGhostStates():
        distToAllGhost.append(manhattanDistance(pacmanPos, ghost.getPosition()))
        if manhattanDistance(pacmanPos, ghost.getPosition()) <= 3:
                stateScore -= 11
    
    #calculate the distance to all the food
    distToAllFood = [0]
    foodCounter = 0
    for food in foodMatrix:
        foodCounter += 1
        distToAllFood.append(manhattanDistance(pacmanPos, food))
    
    
    return (currentGameState.getScore()) / (sum(distToAllGhost)+1+min(distToAllFood)) + stateScore   
# Abbreviation
better = betterEvaluationFunction
