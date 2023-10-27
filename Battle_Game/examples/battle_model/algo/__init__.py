from . import ac
from . import q_learning

AC = ac.ActorCritic
MFAC = ac.MFAC
IL = q_learning.DQN
MFQ = q_learning.MFQ


def spawn_ai(algo_name, sess, env, handle, human_name, max_steps, wtoe):
    if algo_name == 'mfq':
        model = MFQ(sess, human_name, handle, env, max_steps, memory_size=80000, wtoe=wtoe)
    elif algo_name == 'mfac':
        model = MFAC(sess, human_name, handle, env, wtoe=wtoe)
    elif algo_name == 'ac':
        model = AC(sess, human_name, handle, env, wtoe=wtoe)
    elif algo_name == 'il':
        model = IL(sess, human_name, handle, env, max_steps, memory_size=80000, wtoe=wtoe)
    return model
