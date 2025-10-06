import argparse
from functools import partial
from time import sleep

from stable_baselines3 import DQN, PPO

from envs.connect4 import _connect4

usage = "\narguments:\n" + \
        "-h: help\n" + \
        "--algo [PPO, DQN]: algorithm to be used, defaults to DQN\n" + \
        "--model-file [<some_model_file.zip>]: model file of the agent\n" + \
        "--n-games [<integer number of games>]: defaults to 1\n" + \
        "--first [human, agent]: human or agent first, defaults to agent"

parser = argparse.ArgumentParser(usage=usage,
                                 description="Play Connect4 against a trained agent.",
                                 epilog="Example of usage:\n"
                                        "python play_connect4.py --algo PPO --model-file model_1000000_steps.zip "
                                        "--n-games 1 --first agent")

parser.add_argument("--algo", default="DQN", choices=["PPO", "DQN"])
parser.add_argument("--model-file", required=True)
parser.add_argument("--n-games", default=1, type=int)
parser.add_argument("--first", default="agent", choices=["agent", "human"])

if __name__ == '__main__':
    def play_agent(player: str):
        done = False

        sleep(2)
        agent_action = model.predict(aec_env.states[player][1], deterministic=True)[0].item()
        print("\nAgent plays: " + str(agent_action + 1))
        aec_env._update_state(player, agent_action)
        aec_env.render(agent=player)

        if aec_env._check_win(aec_env.states[player][1, 0]):
            print("Agent wins!")
            done = True
        elif (aec_env.states["player_0"][1] > 0).all():
            print("Game ends in draw.")
            done = True

        return done


    def play_human(player: str):
        done = False

        while True:
            human_action = input("\nHuman plays: ")
            try:
                human_action = int(human_action) - 1
            except:
                print("Action should be an integer between 1 and " + str(aec_env.width))
                continue
            if human_action > (aec_env.width - 1) or human_action < 0:
                print("Action should be between 1 and " + str(aec_env.width))
                continue
            break
        aec_env._update_state(player, human_action)
        aec_env.render(agent=player)

        if aec_env._check_win(aec_env.states[player][1, 0]):
            print("Human wins!")
            done = True
        elif (aec_env.states[player][1] > 0).all():
            print("Game ends in draw.")
            done = True

        return done


    args = parser.parse_args()

    n_games = args.n_games
    algorithm_cls = eval(args.algo)
    model_file = args.model_file
    first = args.first

    if first == "agent":
        plays = [partial(play_agent, "player_0"), partial(play_human, "player_1")]
        start_msg = "Agent goes first as P1, human goes second as P2"
    else:
        plays = [partial(play_human, "player_0"), partial(play_agent, "player_1")]
        start_msg = "Human goes first as P1, agent goes second as P2"

    aec_env = _connect4()

    custom_objects = {'n_envs': 1}
    model = algorithm_cls.load(model_file, device="cpu", custom_objects=custom_objects)

    print(start_msg)

    for n in range(n_games):
        aec_env.reset()
        print("Game " + str(n + 1) + "/" + str(n_games) + " begins!")
        sleep(2)
        aec_env.render()

        while True:
            if plays[0]():
                break
            if plays[1]():
                break
