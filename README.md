How to run the files:

The multiAgents.py file contains our MCTS Agents. 
There are 2 MCTS Agents. MCTSAgentA for part A of the project. MCTSAgentB for part B of the project.
In MCTSAgentB for part B, there is also a line commented for choosing randomAction as required in the question. 
This file also has other agents like Minimax, Expectimax, AlphaBeta and Reflex Agent.
In order to run the Search Agent, we have to run the command

python pacman.py --frameTime 0 -p "AGENT NAME"  -l "LAYOUT NAME" 

This is will run the Desired agent on the desired layout for 1 run.

- Available Agents 
1) ExpectimaxAgent
2) MinimaxAgent
3) AlphaBetaAgent
4) MCTSAgentA
5) MCTSAgentB 

- Available Layouts
1) capsuleClassic
2) contestClassic
3) mediumClassic
4) minimaxClassic
5) openClassic
6) originalClassic
7) powerClassic
8) smallClassic
9) testClassic
10) trappedClassic
11) trickyClassic


The t_test.py file contains the script to perform the Paired Student T-Test.
It uses the Shapiro-Wilk test to verify the normality of the score distributions and then conduct paired Student’s T-tests. Based on the computational resources available we can change the number of runs to 100. 
To run t_test.py
python t_test.py




