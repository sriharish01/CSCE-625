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

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
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

    def evaluationFunction(self, currentGameState, action):
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
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        newFood = newFood.asList()
        ghostPos = [(G.getPosition()[0], G.getPosition()[1]) for G in newGhostStates]
        scared = min(newScaredTimes) > 0

        # if not new ScaredTimes new state is ghost: return lowest value

        if not scared and (newPos in ghostPos):
            return -1.0

        if newPos in currentGameState.getFood().asList():
            return 1

        closestFoodDist = sorted(newFood, key=lambda fDist: util.manhattanDistance(fDist, newPos))
        closestGhostDist = sorted(ghostPos, key=lambda gDist: util.manhattanDistance(gDist, newPos))

        fd = lambda fDis: util.manhattanDistance(fDis, newPos)
        gd = lambda gDis: util.manhattanDistance(gDis, newPos)

        return 1.0 / fd(closestFoodDist[0]) - 1.0 / gd(closestGhostDist[0])

def scoreEvaluationFunction(currentGameState):
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
    def terminate(self, state, d):
            return state.isWin() or state.isLose() or d == self.depth


    def min_value(self, state, d, ghost):  # minimizer

        if self.terminate(state, d):
            return [Directions.STOP, self.evaluationFunction(state)]

        v = float('inf')
        for action in state.getLegalActions(ghost):
            if ghost == state.getNumAgents() - 1:
                new_score = self.max_value(state.generateSuccessor(ghost, action), d + 1)[1]
                if new_score < v:
                    v = new_score
                    chosenAction = action
            else:
                new_score = self.min_value(state.generateSuccessor(ghost, action), d, ghost + 1)[1]
                if new_score < v:
                    v = new_score
                    chosenAction = action
        return [chosenAction, v]

    def max_value(self, state, d):  # maximizer

        if self.terminate(state, d):
            return [Directions.STOP, self.evaluationFunction(state)]

        v = float('-inf')
        for action in state.getLegalActions(0):
            new_score =  self.min_value(state.generateSuccessor(0, action), d, 1)[1]
            if new_score > v:
                v = new_score
                chosenAction = action

        return [chosenAction, v]


    def getAction(self, gameState):
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


        return self.max_value(gameState, 0)[0]


        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def terminate(self, state, d):
        return state.isWin() or state.isLose() or d == self.depth


    def min_value(self, state, d, ghost, A, B):  # minimizer

        if self.terminate(state, d):
            return [Directions.STOP, self.evaluationFunction(state)]

        v = float('inf')
        for action in state.getLegalActions(ghost):
            if ghost == state.getNumAgents() - 1:
                new_score = self.max_value(state.generateSuccessor(ghost, action), d + 1, A, B)[1]
                if new_score < v:
                    v = new_score
                    chosenAction = action
            else:
                new_score = self.min_value(state.generateSuccessor(ghost, action), d, ghost + 1, A, B)[1]
                if new_score < v:
                    v = new_score
                    chosenAction = action

            if v < A:
                return [chosenAction, v]    

            B = min(B, v)

        return [chosenAction, v]

    def max_value(self, state, d, A = float('-inf'), B= float('inf')):  # maximizer

        if self.terminate(state, d):
            return [Directions.STOP, self.evaluationFunction(state)]

        v = float('-inf')
        for action in state.getLegalActions(0):
            new_score =  self.min_value(state.generateSuccessor(0, action), d, 1, A, B)[1]
            if new_score > v:
                v = new_score
                chosenAction = action

            if v > B:
                return [chosenAction, v]    

            A = max(A, v)

        return [chosenAction, v]

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        return self.max_value(gameState,0)[0]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def terminate(self, state, d):
            return state.isWin() or state.isLose() or d == self.depth


    def expect_value(self, state, d, ghost):  # minimizer

        if self.terminate(state, d):
            return [Directions.STOP, self.evaluationFunction(state)]

        v = float('inf')
        if ghost == state.getNumAgents() - 1:
            scores = [ self.max_value(state.generateSuccessor(ghost, action), d + 1)[1] for action in state.getLegalActions(ghost)]
        else:
            scores = [ self.expect_value(state.generateSuccessor(ghost, action), d, ghost + 1)[1] for action in state.getLegalActions(ghost)]
        v = sum(scores) / float(len(scores)) 
        chosenAction = Directions.STOP   

        return [chosenAction, v]

    def max_value(self, state, d):  # maximizer

        if self.terminate(state, d):
            return [Directions.STOP, self.evaluationFunction(state)]
        chosenAction = Directions.STOP
        v = float('-inf')
        for action in state.getLegalActions(0):
            new_score =  self.expect_value(state.generateSuccessor(0, action), d, 1)[1]
            if new_score > v:
                v = new_score
                chosenAction = action

        return [chosenAction, v]

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.max_value(gameState,0)[0]

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: Minimize the distance to the closest food and maximize the distance to the closest ghost.

    Increase the variation of the utility if the ghost is less than 3 units away.

    Maximize the uitility for eating power capsules
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newGhostPos = currentGameState.getGhostPositions()
    newFood = newFood.asList()

    nearestGhostDistance = 1000
    for ghostState in newGhostStates:
        if ghostState.scaredTimer == 0:
            nearestGhostDistance = min(nearestGhostDistance, util.manhattanDistance(ghostState.configuration.pos, newPos))

    nearestFoodDistance = 1000
    averageDistance = 0.0
    for food in newFood:
        distance = util.manhattanDistance(food, newPos)
        averageDistance += distance
        nearestFoodDistance = min(nearestFoodDistance, distance)

    averageDistance = averageDistance / float(len(newFood)+0.01)

    if currentGameState.isLose():
        return float('-inf')

    if newPos in newGhostPos:
        return float('-inf')

    score = 0

    if nearestGhostDistance <3:
        score-=300
    if nearestGhostDistance <2:
        score-=1000
    if len(currentGameState.getCapsules()) < 2:
        score+=100
    if len(currentGameState.getCapsules()) < 1:
        score+=300
            
    if len(newFood)==0 or len(newGhostStates)==0 :
        score += currentGameState.getScore() 
    else:
        score += currentGameState.getScore() - (0.9 * nearestFoodDistance + 0.1 * averageDistance)  - 1.0 / (nearestGhostDistance)
    
    return score

# Abbreviation
better = betterEvaluationFunction
