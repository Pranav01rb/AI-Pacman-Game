import numpy as np
from scipy.stats import ttest_rel, shapiro
from pacman import runGames, readCommand
from multiAgents import MinimaxAgent, MCTSAgent, MCTSAgentB #Import agent name
import textDisplay


def check_normality(scores):
    stat, p_value = shapiro(scores)
    print(f"Normality test for scores: stat={stat}, p_value={p_value}")
    return p_value 

def perform_t_test(scores1, scores2):
    stat, p_value = ttest_rel(scores1, scores2)
    print(f"Paired T-test: stat={stat}, p_value={p_value}")
    return stat, p_value


def simulate_agent(agent_class, layout, num_simulations, depth=None):
    scores = []
    for _ in range(num_simulations):

        args_str = [
            '-p', agent_class.__name__,
            '-l', layout,
            '--frameTime', '0',
            '-q', 
            '-n', '1', 
            '--timeout', '1'
        ]
        

        if depth is not None and issubclass(agent_class, MinimaxAgent):
            args_str += ['-a', f'depth={depth}']
        
        args = readCommand(args_str)
        args['display'] = textDisplay.NullGraphics() 
        args['pacman'] = agent_class()
        
        # Run the game
        games = runGames(**args)
        
        # Store the score
        scores.append(games[0].state.getScore())
    print(scores)
    return scores

minimax_depth = 4
num_simulations = 50
layouts = ['testClassic','smallClassic','mediumClassic'] 
for layout in layouts:
    print(f"Running simulations for layout: {layout}")
    minimax_scores = simulate_agent(MinimaxAgent, layout, num_simulations, depth=minimax_depth)
    mcts_scores = simulate_agent(MCTSAgent, layout, num_simulations)
    mctsb_scores = simulate_agent(MCTSAgentB, layout, num_simulations) 
    print(" ")
    print(minimax_scores)
    print(mcts_scores)
    print(mctsb_scores) # Change to proper agent
    print(" ")
    print(f"Layout = {layout}")
    print("Number of runs = 50")
    print(" ")
    print("Normality Test for Minimax Agent:::")
    a = check_normality(minimax_scores)
    print("Normality Test for MCTS Agent A:::")
    b = check_normality(mcts_scores)
    print("Normality Test for MCTS Agent B:::")
    c = check_normality(mctsb_scores)
    print(" ")
    print("p_value for Minimax Agent:::")
    print(a)
    print("p_value for MCTS Agent A:::")
    print(b)
    print("p_value for MCTS Agent B:::")
    print(c)
    print(" ")
    
    if a > 0.05 and b > 0.05:
        print("Comparing MinimaxAgent with MCTSAgent")
        perform_t_test(minimax_scores, mcts_scores)
    else:
        print("Normality assumption not met for MinimaxAgent and MCTSAgent")

    if a > 0.05 and c > 0.05:
        print("Comparing MinimaxAgent with MCTSAgentb")
        perform_t_test(minimax_scores, mctsb_scores)
    else:
        print("Normality assumption not met for MinimaxAgent and MCTSAgentB")
        
