import argparse

"""
Here are the param for the training

"""

def get_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    parser.add_argument("--algorithm", type=str, default="WToE")
    parser.add_argument("--scenario-name", type=str, default="simple_spread", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=100, help="maximum episode length")
    parser.add_argument("--time-steps", type=int, default=1000000, help="number of time steps")
    parser.add_argument("--num-adversaries", type=int, default=1, help="number of adversaries")
    parser.add_argument("--level_k", type=int, default=5)
    parser.add_argument("--lr-actor", type=float, default=1e-4, help="learning rate of actor")
    parser.add_argument("--lr-critic", type=float, default=1e-3, help="learning rate of critic")
    parser.add_argument("--lr-vae", type=float, default=1e-3, help="learning rate of vae")
    parser.add_argument("--epsilon", type=float, default=0.1, help="epsilon greedy")
    parser.add_argument("--noise_rate", type=float, default=0.1, help="noise rate for sampling from a standard normal distribution ")

    parser.add_argument("--bvae_rate_inital", type=float, default=0.2)
    parser.add_argument("--bvae_rate_final", type=float, default=0.01)

    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--tau", type=float, default=0.01, help="parameter for updating the target network")
    parser.add_argument("--buffer_size", type=int, default=int(5e5), help="number of transitions can be stored in buffer")
    parser.add_argument("--batch_size", type=int, default=64, help="number of episodes to optimize at the same time")
    parser.add_argument("--history_size", type=int, default=32, help="number of history transitions to optimize at the same time")
    parser.add_argument("--layers_before_gru", nargs='+', type=int, default=[])
    parser.add_argument("--layers_after_gru", nargs='+', type=int, default=[])
    parser.add_argument("--gru_hidden_size", type=int, default = 64)
    parser.add_argument("--latent_dim", type = int, default = 32)
    parser.add_argument("--state_embedding_size", type=int, default=10)
    parser.add_argument("--action_embedding_size", type=int, default=10)
    parser.add_argument("--reward_embedding_size", type=int, default=5)
    parser.add_argument("--decoder_layers", nargs='+', type=int, default=[32])

    parser.add_argument("--save-dir", type=str, default="./log", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=2000, help="save model once every time this many episodes are completed")
    parser.add_argument("--model-dir", type=str, default="", help="directory in which training state and model are loaded")

    parser.add_argument("--evaluate-episodes", type=int, default=10, help="number of episodes for evaluating")
    parser.add_argument("--evaluate-episode-len", type=int, default=100, help="length of episodes for evaluating")
    parser.add_argument("--evaluate", type=bool, default=False, help="whether to evaluate the model")
    parser.add_argument("--evaluate-rate", type=int, default=1000, help="how often to evaluate model")
    parser.add_argument("--cuda", type=bool, default=False)
    args = parser.parse_args()

    return args
