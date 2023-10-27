from common.arguments import get_args
from common.utils import make_env

if __name__ == '__main__':
    # get the params
    args = get_args()
    env, args = make_env(args)
    if args.algorithm == "WToE":
        from Runner.runner_WToE import Runner
    elif args.algorithm == "PR2":
        from Runner.runner_pr2 import Runner
    elif args.algorithm == "NoisyNet":
        from Runner.runner_noisynet import Runner
    elif args.algorithm == "WToE_MADDPG":
        from Runner.runner_wtoe_maddpg import Runner
    elif args.algorithm == "WToE_NOISE":
        from Runner.runner_wtoe_noisenet import Runner
    elif args.algorithm == "MADDPG_WToE":
        from Runner.runner_maddpg_wtoe import Runner
    elif args.algorithm == "MADDPG_NOISE":
        from Runner.runner_maddpg_noisenet import Runner
    elif args.algorithm == "NOISE_WToE":
        from Runner.runner_noisenet_wtoe import Runner
    elif args.algorithm == "NOISE_MADDPG":
        from Runner.runner_noisenet_maddpg import Runner
    elif args.algorithm == "PR2_MADDPG":
        from Runner.runner_pr2_maddpg import Runner
    elif args.algorithm == "PR2_NOISE":
        from Runner.runner_pr2_noisynet import Runner
    elif args.algorithm == "PR2_WToE":
        from Runner.runner_pr2_wtoe import Runner
    else:
        from Runner.runner_maddpg import Runner
    runner = Runner(args, env)
    if args.evaluate:
        returns = runner.evaluate()
        print('Average returns is', returns)
    else:
        runner.run()
