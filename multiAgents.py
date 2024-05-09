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
import math
from math import sqrt, log
from pacman import GameState
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
        # print("legal Moves: ", legalMoves)
        # print("Game state: ")
        # print(gameState)

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best
        # print("Action taken by the Pacman: ", legalMoves[chosenIndex])
        # print(".....................")

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
        # print("New Pacman position: ", newPos)
        newFood = successorGameState.getFood()
        # print("Remaining food: ", newFood.asList())
        newGhostStates = successorGameState.getGhostStates()
        # print("New ghost states: ", newGhostStates)
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        # print("Number of moves ghost remains scared: ", newScaredTimes)
        
        "*** YOUR CODE HERE ***"

        newGhostPos = successorGameState.getGhostPosition(1)
        # print("Ghost position", newGhostPos)
        evalValue = 0
        for food in newFood.asList():
            pacmanGhostDistance = manhattanDistance(newPos, newGhostPos)
            pacmanFoodDistance = manhattanDistance(newPos, food)
            ghostFoodDistance = manhattanDistance(newGhostPos, food)
            # evalValue += 1/pacmanFoodDistance
            evalValue += (pacmanGhostDistance/3)/pacmanFoodDistance
        return evalValue + successorGameState.getScore()
        # return evalValue
    
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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        numOfAgents = gameState.getNumAgents()
        numOfGhosts = numOfAgents - 1

        def minimaxValue(gameState, agentIndex, depth):
            if depth==self.depth or gameState.isWin() or gameState.isLose():
                return scoreEvaluationFunction(gameState)
            if agentIndex == 0:
                return maxValue(gameState, depth)
            else:
                return minValue(gameState, agentIndex, depth)
            
        def maxValue(gameState, depth):
            v = float("-inf")
            legalMovesPacman = gameState.getLegalActions(0)
            for action in legalMovesPacman:
                successor = gameState.generateSuccessor(0, action)
                successorValue = minimaxValue(successor, 1, depth)
                v = max(v, successorValue)
            return v
        
        def minValue(gameState, agentIndex, depth):
            v = float("+inf")
            legalMovesGhost = gameState.getLegalActions(agentIndex)
            for action in legalMovesGhost:
                successor = gameState.generateSuccessor(agentIndex, action)
                if agentIndex != numOfGhosts:
                    successorValue = minimaxValue(successor, agentIndex+1, depth)
                else:
                    successorValue = minimaxValue(successor, 0, depth+1)
                v = min(v, successorValue)
            return v
        
        legalMovesPacman = gameState.getLegalActions(0)
        # print("Legal moves of Pacman: ", legalMovesPacman)
        rootNodeSuccessorValues = []
        for action in legalMovesPacman:
            successor = gameState.generateSuccessor(0, action)
            # print("Successors of root node: ")
            # print(successor)
            rootNodeSuccessorValues += [minimaxValue(successor, 1, 0)]
        # print("List of the values of the successors of the root node: ", rootNodeSuccessorValues)
        optimalValue = max(rootNodeSuccessorValues)
        # print("Optimal value: ", optimalValue)
        for index in range(len(rootNodeSuccessorValues)):
            if rootNodeSuccessorValues[index] == optimalValue:
                optimalIndex = index
        optimalAction = legalMovesPacman[optimalIndex]

        # return minimaxValue(gameState, 0, 0)
        return optimalAction
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
        
        numOfAgents = gameState.getNumAgents()
        numOfGhosts = numOfAgents - 1
        initialAlphaValue = float("-inf")
        initialBetaValue = float("+inf")
        # valueAction = {float("-inf"): 'Stop'}

        def minimaxValueAction(gameState, agentIndex, depth, alpha, beta):
            if depth==self.depth or gameState.isWin() or gameState.isLose():
                return scoreEvaluationFunction(gameState), 'Stop'
            if agentIndex == 0:
                return maxValueAction(gameState, depth, alpha, beta)
            else:
                return minValueAction(gameState, agentIndex, depth, alpha, beta)
            
        def maxValueAction(gameState, depth, alpha, beta):
            v = float("-inf")
            a = 'Stop'
            legalMovesPacman = gameState.getLegalActions(0)
            for action in legalMovesPacman:
                # valueAction[v] = action
                successor = gameState.generateSuccessor(0, action)
                # print("Agent index: Depth: ", 0, depth)
                successorValue, _ = minimaxValueAction(successor, 1, depth, alpha, beta)
                v = max(v, successorValue)
                if v == successorValue:
                    a = action
                if v > beta:
                    # print("v value: ", v)
                    # print("a action: ", a)
                    return v, a
                alpha = max(alpha, v)
                # print("Alpha value: ", alpha)
                # print("v value: ", v)
                # print("a action: ", a)
            return v, a
        
        def minValueAction(gameState, agentIndex, depth, alpha, beta):
            v = float("+inf")
            a = 'Stop'
            legalMovesGhost = gameState.getLegalActions(agentIndex)
            for action in legalMovesGhost:
                # valueAction[v] = action
                successor = gameState.generateSuccessor(agentIndex, action)
                # print("Agent index: Depth: ", agentIndex, depth)
                if agentIndex != numOfGhosts:
                    successorValue, _ = minimaxValueAction(successor, agentIndex+1, depth, alpha, beta)
                else:
                    successorValue, _ = minimaxValueAction(successor, 0, depth+1, alpha, beta)
                v = min(v, successorValue)
                if v == successorValue:
                    a = action
                if v < alpha:
                    # print("v value: ", v)
                    # print("a action: ", a)
                    return v, a
                beta = min(beta, v)
                # print("Beta value: ", beta)
                # print("v value: ", v)
                # print("a action: ", a)
            return v, a

        optimalValue, optimalAction = minimaxValueAction(gameState, 0, 0, initialAlphaValue, initialBetaValue)
        # print("Optimal value: ", optimalValue)
        # print("Optimal action: ", optimalAction)
        return optimalAction
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
        numOfAgents = gameState.getNumAgents()
        numOfGhosts = numOfAgents - 1

        def expectimaxValue(gameState, agentIndex, depth):
            if depth==self.depth or gameState.isWin() or gameState.isLose():
                return scoreEvaluationFunction(gameState)
            if agentIndex == 0:
                return maxValue(gameState, depth)
            else:
                return expValue(gameState, agentIndex, depth)
            
        def maxValue(gameState, depth):
            v = float("-inf")
            legalMovesPacman = gameState.getLegalActions(0)
            for action in legalMovesPacman:
                successor = gameState.generateSuccessor(0, action)
                successorValue = expectimaxValue(successor, 1, depth)
                v = max(v, successorValue)
                # print("v value: ", v)
            return v
        
        def expValue(gameState, agentIndex, depth):
            v = 0
            legalMovesGhost = gameState.getLegalActions(agentIndex)
            # print("Legal moves of ghost: ", legalMovesGhost)
            for action in legalMovesGhost:
                successor = gameState.generateSuccessor(agentIndex, action)
                if agentIndex != numOfGhosts:
                    successorValue = expectimaxValue(successor, agentIndex+1, depth)
                else:
                    successorValue = expectimaxValue(successor, 0, depth+1)
                p = 1 / len(legalMovesGhost)
                # print("Probability value: ", p)
                v += p * successorValue
                # print("v value: ", v)
            return v
        
        legalMovesPacman = gameState.getLegalActions(0)
        # print("Legal moves of Pacman: ", legalMovesPacman)
        rootNodeSuccessorValues = []
        for action in legalMovesPacman:
            successor = gameState.generateSuccessor(0, action)
            # print("Successors of root node: ")
            # print(successor)
            rootNodeSuccessorValues += [expectimaxValue(successor, 1, 0)]
        # print("List of the values of the successors of the root node: ", rootNodeSuccessorValues)
        optimalValue = max(rootNodeSuccessorValues)
        # print("Optimal value: ", optimalValue)
        for index in range(len(rootNodeSuccessorValues)):
            if rootNodeSuccessorValues[index] == optimalValue:
                optimalIndex = index
        optimalAction = legalMovesPacman[optimalIndex]
        # print("Optimal action: ", optimalAction)


        # return expectimaxValue(gameState, 0, 0)
        return optimalAction
        util.raiseNotDefined()


###### Project 2 - Game ######

class Node:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.visits = 0
        self.value = 0
        self.children = {}  

class MCTSAgentA(Agent):
    def __init__(self, exploration_constant=0.707):
        self.num_simulations = 25
        self.exploration_constant = exploration_constant
        self.scores = []
        
    def getAction(self, gameState):
        root_node = Node(gameState)
        for i in range(self.num_simulations):
            selected_node = self.select(root_node)
            new_node = self.expand(selected_node)
            simulation_result = self.simulate(new_node.state)
            self.backpropagate(new_node, simulation_result)
        #print(root_node.children)
        best_child = None
        for child in root_node.children.values():
            if best_child is None or child.visits >= best_child.visits:
                best_child = child
        return best_child.action

    def ucb(self, node, parent_visits):
        if node.visits == 0:
            return float('inf')
        exploration_term = 2* self.exploration_constant * math.sqrt(2 * math.log(parent_visits) / node.visits)
        return (node.value / node.visits) + exploration_term

    def select(self, node):
        while node.children:
            maxUcb = float('-inf')
            selectedAction = None
            for action in node.children:
                ucbValue = self.ucb(node.children[action],node.visits)
                if ucbValue > maxUcb:
                    selectedAction = action
                    maxUcb = ucbValue
            node = node.children[selectedAction]
        return node

    def expand(self, node):
        legal_actions = node.state.getLegalActions(0)
        if not legal_actions:
            return node
        for action in legal_actions:
            successor_state = node.state.generateSuccessor(0, action)
            new_node = Node(successor_state, node, action)
            node.children[action] = new_node  
        return new_node
    

    def simulate(self, state):
        while True:
            loop = not state.isWin() and not state.isLose()
            if loop:
                action = random.choice(state.getLegalActions(0))
                state = state.generateSuccessor(0, action)
            else:
                break
        return state.getScore()

    def backpropagate(self, node, value):
        while node:
            node.visits += 1
            node.value += value
            node = node.parent


class Node2:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.visits = 0
        self.value = 0
        self.rave_visits = 0
        self.rave_value = 0
        self.children = {}



class MCTSAgentB(Agent):
    def __init__(self, exploration_constant=2, rave_constant=1000):
        self.num_simulations = 25
        self.exploration_constant = exploration_constant
        self.rave_constant = rave_constant
        self.transposition_table = {}

    def getAction(self, gameState):
        root_node = self.lookup_transposition_table(gameState)
        if root_node is None:
            root_node = Node2(gameState)
        
        for _ in range(self.num_simulations):
            node = self.select(root_node)
            child_node = self.expand(node)
            reward = self.simulate(child_node)
            self.backpropagate(child_node, reward)
            best_child = None
            for child in root_node.children.values():
                if best_child is None or child.visits >= best_child.visits:
                    best_child = child

        self.store_transposition_table(gameState, root_node)
        return best_child.action

    def select(self, node):
        while node.children: 
            max_ucb_rave_value = float('-inf')
            best_child_node = None        
            for child in node.children.values():
                ucb_rave_value = self.ucb_rave(child, node)
                if ucb_rave_value > max_ucb_rave_value:
                    max_ucb_rave_value = ucb_rave_value
                    best_child_node = child
            node = best_child_node
        return node

    def ucb_rave(self, node, parent):
        if node.visits == 0:
            return float('inf')
        
        ucb = node.value / node.visits + self.exploration_constant * math.sqrt(math.log(parent.visits) / node.visits)
        if node.rave_visits > 0:
            rave = node.rave_value / node.rave_visits
        else:
            rave = 0
        alpha = max(0, (self.rave_constant - node.visits) / self.rave_constant)
        return (1 - alpha) * ucb + alpha * rave

    def expand(self, node):
        untried_actions = []
        for action in node.state.getLegalActions(0):
            if action not in node.children:
                untried_actions.append(action)
        if not untried_actions:
            return node
        
        parentnode = node
        for action in untried_actions:
            if action == 'Stop':
                continue
            child_state = parentnode.state.generateSuccessor(0, action)
            child_node = Node2(child_state, parentnode, action)
            parentnode.children[action] = child_node
        return child_node

    def simulate(self, node):
        state = node.state
        self.visited_actions = set()
        count = 0
        time_penalty = -5
        completion_bonus = 10
        repetition_threshold = 4
        repetition_penalty = -100000
        death_penalty = -200000
        recent_actions = []

        while not state.isWin() and not state.isLose() and count < 10:
            count += 1
            legal_actions = state.getLegalActions(0)
            action_weights = self.calculate_action_weights(state, legal_actions)
            idx = action_weights.index(max(action_weights))
            action = legal_actions[idx]
            if action == 'Stop':
                continue
            state = state.generateSuccessor(0, action)
            self.visited_actions.add(action)

            recent_actions.append(action)
            if len(recent_actions) >= repetition_threshold:
                if self.is_repetitive_motion(recent_actions[-repetition_threshold:]):
                    reward = state.getScore() + repetition_penalty
                    break

        if state.isWin():
            reward = state.getScore() + completion_bonus
        elif state.isLose():
            reward = state.getScore() + death_penalty
        else:
            reward = state.getScore()
        reward += count * time_penalty * 0.8
        return reward
    
    def is_repetitive_motion(self, actions):
        if len(actions) < 4:
            return False

        for i in range(len(actions) - 3):
            if actions[i] == actions[i+2] and actions[i+1] == actions[i+3]:
                return True
        return False

    def calculate_action_weights(self, state, legal_actions):
        weights = []
        for action in legal_actions:
            successor_state = state.generateSuccessor(0, action)
            weight = self.evaluate_state(successor_state)
            weights.append(weight)
        return weights

    def evaluate_state(self, state):
        pacman_pos = state.getPacmanPosition()
        ghost_positions = state.getGhostPositions()
        ghost_states = state.getGhostStates()
        food_list = state.getFood().asList()
        capsules = state.getCapsules()

        pacman_powered_up = []
        for ghost in ghost_states:
            if ghost.scaredTimer > 0:
                pacman_powered_up.append(True)
            else:
                pacman_powered_up.append(False)
        pacman_powered_up = any(pacman_powered_up)

        if pacman_powered_up:
            nearest_ghost_distance = []
            for ghost_pos in ghost_positions:
                nearest_ghost_distance.append(manhattanDistance(pacman_pos, ghost_pos))
            nearest_ghost_distance = min(nearest_ghost_distance)
            if(capsules):
                nearest_capsule_distance = []
                for capsule_pos in capsules:
                    nearest_capsule_distance.append(manhattanDistance(pacman_pos, capsule_pos))
                nearest_capsule_distance = min(nearest_capsule_distance)
            else:
                nearest_capsule_distance = 0

            ghost_weight = -5.0 / (nearest_ghost_distance + 1)
            capsule_weight = 5.0 / (nearest_capsule_distance + 1)

        else:
            nearest_ghost_distance = []
            for ghost_pos in ghost_positions:
                nearest_ghost_distance.append(manhattanDistance(pacman_pos, ghost_pos))
            nearest_ghost_distance = min(nearest_ghost_distance)
            if(capsules):
                nearest_capsule_distance = []
                for capsule_pos in capsules:
                    nearest_capsule_distance.append(manhattanDistance(pacman_pos, capsule_pos))
                nearest_capsule_distance = min(nearest_capsule_distance)
            else:
                nearest_capsule_distance = 0

            ghost_weight = 1.0 / (nearest_ghost_distance + 1)
            capsule_weight = 10.0 / (nearest_capsule_distance + 1)

        if(food_list):
            nearest_food_distance = []
            for food_pos in food_list:
                nearest_food_distance.append(manhattanDistance(pacman_pos, food_pos))
            nearest_food_distance = min(nearest_food_distance)
        else:
            nearest_food_distance = 0

        food_weight = 5.0 / (nearest_food_distance + 1)

        total_weight = food_weight + capsule_weight - ghost_weight
        return total_weight

    def backpropagate(self, node, value):
        while node:
            node.visits += 1
            node.value += value
            node.rave_visits += 1
            if node.action in self.visited_actions:
                node.rave_value += value
            node = node.parent

    def lookup_transposition_table(self, state):
        return self.transposition_table.get(self.state_key(state))

    def store_transposition_table(self, state, node):
        self.transposition_table[self.state_key(state)] = node

    def state_key(self, state):
        return tuple(state.getPacmanPosition()), tuple(state.getFood().asList()), tuple(state.getCapsules()), tuple(state.getGhostPositions())