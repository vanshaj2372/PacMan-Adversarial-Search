# Christopher Oliver
# CS5100 - Foundations of Artificial Intelligence
# Programming Assignment 2
# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util
callCount = 0

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
        some Directions.X for some X in the set {North, South, West, East, Stop}
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
        # Stay away from the ghosts!  Enforce a large penalty to the evaluation
        # of a move if it brings Pacman close to a ghost.
        ghostComponent = 0
        for newGhosts in newGhostStates:
            if util.manhattanDistance(newPos,newGhosts.getPosition()) == 1:
                ghostComponent -= 1000;

        # Get to the food!  Penalize a move if it increases the distance be-
        # tween Pacman and the closest food item.  Augment it if it means food
        # will be eaten.
        foodComponent = 0
        foodDistance = float("inf")
        newFoodList = newFood.asList()
        currentFood = currentGameState.getFood()
        currentFoodList = currentFood.asList()
        
        for newFoodPos in newFoodList:
            newFoodDistance = util.manhattanDistance(newPos,newFoodPos)
            if newFoodDistance < foodDistance:
                foodDistance = newFoodDistance
                    
        if len(newFoodList) < len(currentFoodList):
            foodComponent = 100
        else:
            foodComponent = -1*foodDistance
        
        # Penalize a Stop action but not enough to prevent it from being used
        # to escape a ghost!  This helps avoid Pacman stopping on one side of a
        # wall if a food pellet is just on the other side.
        stopComponent = 0
        if action == "Stop":
            stopComponent = -100

        return successorGameState.getScore() + ghostComponent + foodComponent + stopComponent

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
        """
        "*** YOUR CODE HERE ***"
        bestValue = float("-inf")
        bestAction = None

        for action in gameState.getLegalActions(0):
            nextState = gameState.generateSuccessor(0, action)
            val = minimax(self, nextState, self.depth, False, 1)
            if val > bestValue:
                bestValue = val
                bestAction = action
    
        return bestAction

def minimax(agent, state, depth, isMax, ghostMoving):

    if depth <= 0 or state.isWin() or state.isLose():
        return agent.evaluationFunction(state)

    if isMax:
        bestValue = float("-inf")
        for action in state.getLegalActions(0):
            nextState = state.generateSuccessor(0, action)
            val = minimax(agent, nextState, depth, False, 1)
            bestValue = max(bestValue, val)
        return bestValue
    else:
        bestValue = float("inf")
        for action in state.getLegalActions(ghostMoving):
            nextState = state.generateSuccessor(ghostMoving, action)
            # All the ghosts move one after the other then Pacman moves.
            if ghostMoving >= state.getNumAgents() - 1:
                val = minimax(agent, nextState, depth - 1, True, 1)
            else:
                val = minimax(agent, nextState, depth, False, ghostMoving + 1)
            bestValue = min(bestValue, val)
        return bestValue


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        bestValue = float("-inf")
        val = float("-inf")
        alpha = float("-inf")
        beta = float("inf")
        bestAction = None
        
        for action in gameState.getLegalActions(0):
            nextState = gameState.generateSuccessor(0, action)
            val = alphaBeta(self, nextState, self.depth, False, 1, alpha, beta)
            if val > bestValue:
                bestValue = val
                bestAction = action
            alpha = max(alpha, val)
        
        return bestAction

def alphaBeta(agent, state, depth, isMax, ghostMoving, alpha, beta):
    
    if depth <= 0 or state.isWin() or state.isLose():
        return agent.evaluationFunction(state)
    
    if isMax:
        val = float("-inf")
        for action in state.getLegalActions(0):
            nextState = state.generateSuccessor(0, action)
            val = max(val, alphaBeta(agent, nextState, depth, False, 1, alpha, beta))
            if val > beta:
                return val
            alpha = max(alpha, val)
        return val
    else:
        val = float("inf")
        for action in state.getLegalActions(ghostMoving):
            nextState = state.generateSuccessor(ghostMoving, action)
            # All the ghosts move one after the other then Pacman moves.
            if ghostMoving >= state.getNumAgents() - 1:
                val = min(val, alphaBeta(agent, nextState, depth - 1, True, 1, alpha, beta))
            else:
                val = min(val, alphaBeta(agent, nextState, depth, False, ghostMoving + 1, alpha, beta))
            if val < alpha:
                return val
            beta = min(beta, val)
        return val


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        bestValue = float("-inf")
        bestAction = None
        
        for action in gameState.getLegalActions(0):
            nextState = gameState.generateSuccessor(0, action)
            val = expectimax(self, nextState, self.depth, False, 1)
            if val > bestValue:
                bestValue = val
                bestAction = action
        
        return bestAction

def expectimax(agent, state, depth, isMax, ghostMoving):
    
    if depth <= 0 or state.isWin() or state.isLose():
        return agent.evaluationFunction(state)
    
    if isMax:
        bestValue = float("-inf")
        for action in state.getLegalActions(0):
            nextState = state.generateSuccessor(0, action)
            val = expectimax(agent, nextState, depth, False, 1)
            bestValue = max(bestValue, val)
        return bestValue
    else:
        totalValue = 0.0
        numMoves = len(state.getLegalActions(ghostMoving))
        for action in state.getLegalActions(ghostMoving):
            nextState = state.generateSuccessor(ghostMoving, action)
            # All the ghosts move one after the other then Pacman moves.
            if ghostMoving >= state.getNumAgents() - 1:
                val = expectimax(agent, nextState, depth - 1, True, 1)
            else:
                val = expectimax(agent, nextState, depth, False, ghostMoving + 1)
            totalValue += val
        return totalValue/numMoves

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # Useful variables.
    pacPos = currentGameState.getPacmanPosition()
    capsulePosList = currentGameState.getCapsules()
    walls = currentGameState.getWalls()
    wallsList = walls.asList()
    
    # Deal with the food.
    # - Eat all the food!  The less food left the better the state.
    foodComponent = 0
    foodComponent = currentGameState.getNumFood() * -10
    if foodComponent == 0:
        foodComponent += 1000

    # - Seek the food.
    foodDistance = 0.0
    food = currentGameState.getFood()
    foodList = food.asList()
        
    for foodPos in foodList:
        foodDistance += (util.manhattanDistance(pacPos,foodPos) + 1)

    foodComponent -= foodDistance
    
    # Deal with the ghosts.
    # - Don't get eaten by them!
    # - Eat them when possible!
    ghostComponent = 0
    for ghost in currentGameState.getGhostStates():
        distanceToGhost = util.manhattanDistance(pacPos,ghost.getPosition())
        ghostScared = ghost.scaredTimer > 0
        if ghostScared:
            if distanceToGhost == 0:
                ghostComponent += 200
            else:
                ghostComponent -= distanceToGhost * 100
        else:
            if distanceToGhost == 0:
                ghostComponent -= 1000;

    # Deal with the capsules.
    # - Eat them so the ghosts may be eaten.
    capsuleComponent = len(capsulePosList) * -650

    # Penalize stopping.
    stopComponent = currentGameState.getScore()
    
    # Stay out of the corners unless a capsule may be had.
    wallsAround = 0
    trappedComponent = 0

    if pacPos not in capsulePosList:
        pacX, pacY = pacPos
    #   Check North
        northPos = (pacX, pacY + 1)
        if northPos in wallsList:
            wallsAround += 1
    #   Check South
        southPos = (pacX, pacY - 1)
        if southPos in wallsList:
            wallsAround += 1
    #   Check East
        eastPos = (pacX + 1, pacY)
        if eastPos in wallsList:
            wallsAround += 1
    #   check West
        westPos = (pacX - 1, pacY)
        if westPos in wallsList:
            wallsAround += 1

    if wallsAround == 3:
        trappedComponent = -1000
    

    return foodComponent + ghostComponent + capsuleComponent + stopComponent + trappedComponent

# Abbreviation
better = betterEvaluationFunction

