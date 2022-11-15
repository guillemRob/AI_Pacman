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
import random
import util
import time

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
        scores = [self.evaluationFunction(
            gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(
            len(scores)) if scores[index] == bestScore]
        # Pick randomly among the best
        chosenIndex = random.choice(bestIndices)

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
        newScaredTimes = [
            ghostState.scaredTimer for ghostState in newGhostStates]

        foodGame = currentGameState.getFood().asList()
        maxDistance = -99999
        distance = 0

        # if ghost is close to pacman return low score
        for ghost in newGhostStates:
            if ghost.getPosition() == newPos:
                return -99999

        # return negative distance of closest point to food
        for food in foodGame:
            distance = -(manhattanDistance(food, newPos))
            if (distance > maxDistance):
                maxDistance = distance
        return maxDistance


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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        pacman = 0  # agentIndex = 0

        # following the AIMA 3rd edition

        # for the pacman agent
        def maxValue(gameState, depth):
            currentDepth = depth + 1
            # in terminal state return utility(state)
            if gameState.isWin() or gameState.isLose() or currentDepth == self.depth:
                return self.evaluationFunction(gameState)
            maxval = -99999
            actions = gameState.getLegalActions(pacman)
            for action in actions:
                successor = gameState.generateSuccessor(pacman, action)
                maxval = max(maxval, minValue(successor, currentDepth, 1))
            return maxval

        # for the gohst agents
        def minValue(gameState, depth, agentIndex):
            # in terminal state return utility(state)
            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            minval = 99999
            actions = gameState.getLegalActions(agentIndex)
            for action in actions:
                successor = gameState.generateSuccessor(agentIndex, action)
                if agentIndex == (gameState.getNumAgents() - 1):
                    minval = min(minval, maxValue(successor, depth))
                else:
                    minval = min(minval, minValue(
                        successor, depth, agentIndex+1))

            return minval

        def minmaxDecision(agentIndex, depth):
            actions = gameState.getLegalActions(agentIndex)
            value = -999999
            retAction = None
            for action in actions:
                successor = gameState.generateSuccessor(agentIndex, action)
                scorevalue = minValue(successor, depth, 1)
                # we choose the max value of the successors
                if scorevalue > value:
                    value = scorevalue
                    retAction = action
            return retAction

        return minmaxDecision(pacman, 0)
        util.raiseNotDefined()


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        pacman = 0  # agentIndex = 0

        # following the AIMA 3rd edition

        # for the pacman agent
        def maxValue(gameState, depth, alpha, beta):
            currentDepth = depth + 1
            # in terminal state return utility(state)
            if gameState.isWin() or gameState.isLose() or currentDepth == self.depth:
                return self.evaluationFunction(gameState)
            maxval = -99999
            alphaval = alpha
            actions = gameState.getLegalActions(pacman)
            for action in actions:
                successor = gameState.generateSuccessor(pacman, action)
                maxval = max(maxval, minValue(
                    successor, currentDepth, 1, alphaval, beta))
                if maxval > beta:
                    return maxval
                alphaval = max(alphaval, maxval)
            return maxval

        # for the gohst agents
        def minValue(gameState, depth, agentIndex, alpha, beta):
            # in terminal state return utility(state)
            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            minval = 99999
            betaval = beta
            actions = gameState.getLegalActions(agentIndex)
            for action in actions:
                successor = gameState.generateSuccessor(agentIndex, action)
                if agentIndex == (gameState.getNumAgents() - 1):
                    minval = min(minval, maxValue(
                        successor, depth, alpha, betaval))
                    if minval < alpha:
                        return minval
                    betaval = min(betaval, minval)
                else:
                    minval = min(minval, minValue(
                        successor, depth, agentIndex+1, alpha, betaval))
                    if minval < alpha:
                        return minval
                    betaval = min(betaval, minval)
            return minval

        def alpha_beta(agentIndex, depth):
            actions = gameState.getLegalActions(agentIndex)
            value = -999999999
            alpha = -999999999
            beta = 999999999
            retAction = None
            for action in actions:
                successor = gameState.generateSuccessor(agentIndex, action)
                scorevalue = minValue(successor, depth, 1, alpha, beta)
                # we choose the max value of the successors
                if scorevalue > value:
                    value = scorevalue
                    retAction = action
                if scorevalue > beta:
                    return retAction
                alpha = max(alpha, scorevalue)
            return retAction

        return alpha_beta(pacman, 0)
        util.raiseNotDefined()


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
        util.raiseNotDefined()


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
